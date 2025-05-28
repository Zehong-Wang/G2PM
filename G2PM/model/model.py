import copy
import os.path as osp
import random
import numpy as np
import torch
import torch_scatter
import torch.nn as nn
import torch.nn.functional as F
# from torch_geometric.utils import mask_feature

from utils.eval import *
from utils.utils import get_device_from_model, check_path
from .encoder import PatternEncoder
from .vq import VectorQuantize

from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from data.pyg_data_loader import mol_graphs

# Helper functions

def mask_feature(
        x,
        p: float = 0.5,
        mode: str = 'col',
        fill_value: float = 0.,
        training: bool = True,
):
    if p < 0. or p > 1.:
        raise ValueError(f'Masking ratio has to be between 0 and 1 '
                         f'(got {p}')
    if not training or p == 0.0:
        return x, torch.ones_like(x, dtype=torch.bool)
    assert mode in ['row', 'col', 'all']

    if mode == 'row':
        mask = torch.rand(x.size(0), device=x.device) >= p
        mask = mask.view(-1, 1)
    elif mode == 'col':
        mask = torch.rand(x.size(1), device=x.device) >= p
        mask = mask.view(1, -1)
    else:
        mask = torch.rand_like(x) >= p

    x = x.masked_fill(~mask, fill_value)
    return x, mask


def mask_patterns(
        patterns: torch.Tensor,
        p: float = 0.5,
        mode: str = 'mask',  # 'mask' or 'random'
        training: bool = True,
):
    """Mask patterns tensor by either zeroing or randomizing node indices.
    
    Args:
        patterns: Tensor of shape [h, n, k] containing node indices
        p: Probability of masking each position
        mode: 'mask' to mask with -1s, 'random' to replace with random node indices
        training: Whether in training mode
    
    Returns:
        Tuple of (masked patterns, mask boolean tensor)
    """
    if p < 0. or p > 1.:
        raise ValueError(f'Masking ratio has to be between 0 and 1 (got {p})')

    if not training or p == 0.0:
        return patterns, torch.ones_like(patterns, dtype=torch.bool)

    # Create mask of shape [h, n, k]
    mask = torch.rand_like(patterns.float()) >= p

    if mode == 'mask':
        # Mask positions with zeros
        patterns = patterns.masked_fill(~mask, -1)
    elif mode == 'random':
        # Generate random node indices between 0 and n-1
        n = patterns.size(1)
        random_indices = torch.randint_like(patterns, 0, n)
        # Replace masked positions with random indices
        patterns = torch.where(mask, patterns, random_indices)
    else:
        raise ValueError(f"Mode must be 'zero' or 'random', got {mode}")

    return patterns, mask


# Adapted from official ogb implementation
# https://github.com/snap-stanford/ogb/blob/master/examples/linkproppred/ppa/gnn.py
class LinkPredictor(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_dim, hidden_dim))
        self.lins.append(torch.nn.Linear(hidden_dim, output_dim))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x):
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)


class BaseModel(nn.Module):
    def __init__(self, params):
        super(BaseModel, self).__init__()

        self.node_dim = params['input_dim']
        self.edge_dim = params['edge_dim']
        self.input_dim = params['input_dim'] + params['edge_dim'] + params['node_pe_dim']
        self.hidden_dim = params['hidden_dim']
        self.output_dim = params['output_dim']
        self.num_enc_layers = params['num_enc_layers']
        self.num_dec_layers = params['num_dec_layers']
        self.pattern_encoder = PatternEncoder(params)

        self.vq = VectorQuantize(
            dim=self.hidden_dim,
            codebook_size=params["codebook_size"],
            codebook_dim=self.hidden_dim,
            heads=params['num_heads'],
            separate_codebook_per_head=True,
            use_cosine_sim=True,
            kmeans_init=True,
            ema_update=True,
        )

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=params["num_heads"],
            dim_feedforward=self.hidden_dim * 4,
            dropout=params["dropout"],
            norm_first=params["norm_first"]
        )
        self.encoder = nn.ModuleList([copy.deepcopy(self.encoder_layer) for _ in range(self.num_enc_layers)])
        self.encoder_norm = nn.ModuleList([nn.LayerNorm(self.hidden_dim) for _ in range(self.num_enc_layers)])

        if params['task'] in ['node', 'graph']:
            self.head = nn.Linear(self.hidden_dim, self.output_dim)
        elif params['task'] in ['link']:
            self.head = LinkPredictor(self.hidden_dim, self.hidden_dim, 1, 3, 0.0)

        if params['use_cls_token']:
            self.cls_token = nn.Parameter(torch.zeros(1, self.hidden_dim))
            nn.init.normal_(self.cls_token, std=0.02)

        if params['dataset'] in mol_graphs:
            self.atom_encoder = AtomEncoder(emb_dim=100)
            self.bond_encoder = BondEncoder(emb_dim=100)

        self.register_buffer('pre_transformation', None)

    def linear_probe(self):
        for param in self.parameters():
            param.requires_grad = False
        for param in self.head.parameters():
            param.requires_grad = True

    def set_pre_transformation(self, input_dim, output_dim):
        self.pre_transformation = nn.Linear(input_dim, output_dim)

    def reset_head(self, output_dim):
        self.head = nn.Linear(self.hidden_dim, output_dim)

    def transformer_encode(self, x):
        for layer, norm in zip(self.encoder, self.encoder_norm):
            last_x = x
            x = layer(norm(x))
            x = last_x + x
        return x

    def get_instance_emb(self, pattern_emb, params):
        if params['use_cls_token']:
            instance_emb = pattern_emb[0].squeeze(0)
        else:
            instance_emb = pattern_emb.mean(dim=0)
        return instance_emb

    def forward(self, graph, items, params, **kwargs):
        mode = kwargs['mode']
        if params['task'] == 'node':
            return self.encode_node(graph, items, params, mode)
        elif params['task'] == 'link':
            return self.encode_link(graph, items, params, mode)
        elif params['task'] == 'graph':
            return self.encode_graph(graph, items, params, mode)
        else:
            raise ValueError(f"Unsupported task: {params['task']}")

    def encode_node(self, graph, nodes, params, mode):
        device = get_device_from_model(self)

        feat = graph.x
        node_pe = graph.pe if graph.get('pe') is not None else None

        if self.pre_transformation is not None:
            feat = self.pre_transformation(feat)

        # Get patterns
        num_patterns = params['num_patterns']
        pattern_set = params['pattern_set']

        # Get patterns for target nodes
        all_patterns = pattern_set['pattern']
        selected_patterns = all_patterns[:, nodes, :]
        h, num_nodes, k = selected_patterns.shape

        # Randomly select patterns during training
        if mode == 'train':
            idx = torch.randint(0, h, (num_nodes, num_patterns))
            patterns = torch.stack([selected_patterns[idx[i], i, :] for i in range(num_nodes)], dim=1)
        else:
            patterns = selected_patterns

        if graph.edge_attr is not None:
            e_feat = graph.edge_attr
            all_eid = pattern_set['eid']
            selected_eid = all_eid[:, nodes, :]
            if mode == 'train':
                eids = torch.stack([selected_eid[idx[i], i, :] for i in range(num_nodes)], dim=1)
                e_feat = e_feat[eids].to(device)
            else:
                eids = selected_eid
                e_feat = e_feat[eids].to(device)
        else:
            e_feat = None

        pattern_feat = self.pattern_encoder.encode_node(patterns, feat, node_pe, e_feat, params)

        if params['use_vq']:
            pattern_feat, _, commit_loss, _ = self.vq(pattern_feat)
        else:
            commit_loss = 0

        if params['use_cls_token']:
            pattern_feat = torch.cat([self.cls_token.repeat(1, pattern_feat.size(1), 1), pattern_feat], dim=0)
        pattern_emb = self.transformer_encode(pattern_feat)
        instance_emb = self.get_instance_emb(pattern_emb, params)

        pred = self.head(instance_emb)

        return pred, instance_emb, pattern_emb, commit_loss

    def encode_link(self, graph, links, params, mode):
        device = get_device_from_model(self)

        feat = graph.x
        node_pe = graph.pe if graph.get('pe') is not None else None

        source_nodes, target_nodes = links[:, 0], links[:, 1]
        all_nodes = {'source': source_nodes, 'target': target_nodes}
        edge_emb = {'pattern': {}, 'instance': {}, 'commit_loss': {}}

        for key, nodes in all_nodes.items():
            # Get patterns
            num_patterns = params['num_patterns']
            pattern_set = params['pattern_set']

            # Get patterns for target nodes
            all_patterns = pattern_set['pattern']
            selected_patterns = all_patterns[:, nodes, :]
            h, num_nodes, k = selected_patterns.shape

            # Randomly select patterns during training
            if mode == 'train':
                idx = torch.randint(0, h, (num_nodes, num_patterns))
                patterns = torch.stack([selected_patterns[idx[i], i, :] for i in range(num_nodes)], dim=1)
            else:
                patterns = selected_patterns

            if graph.edge_attr is not None:
                e_feat = graph.edge_attr
                all_eid = pattern_set['eid']
                selected_eid = all_eid[:, nodes, :]
                if mode == 'train':
                    eids = torch.stack([selected_eid[idx[i], i, :] for i in range(num_nodes)], dim=1)
                    e_feat = e_feat[eids].to(device)
                else:
                    eids = selected_eid
                    e_feat = e_feat[eids].to(device)
            else:
                e_feat = None

            pattern_feat = self.pattern_encoder.encode_node(patterns, feat, node_pe, e_feat, params)

            if params['use_vq']:
                pattern_feat, _, commit_loss, _ = self.vq(pattern_feat)
            else:
                commit_loss = 0

            pattern_emb = self.transformer_encode(pattern_feat)
            instance_emb = self.get_instance_emb(pattern_emb, params)

            edge_emb['pattern'][key] = pattern_emb
            edge_emb['instance'][key] = instance_emb
            edge_emb['commit_loss'][key] = commit_loss

        instance_emb = edge_emb['instance']['source'] * edge_emb['instance']['target']
        pattern_emb = torch.cat([edge_emb['pattern']['source'], edge_emb['pattern']['target']], dim=-1)

        commit_loss = edge_emb['commit_loss']['source'] + edge_emb['commit_loss']['target']

        pred = self.head(instance_emb)

        return pred, instance_emb, pattern_emb, commit_loss

    def encode_graph(self, graph, graphs, params, mode):
        device = get_device_from_model(self)

        feat = graph._data.x_feat.to(device)

        # Get patterns
        num_patterns = params['num_patterns']
        pattern_set = params['pattern_set']
        if pattern_set.get('train') is not None:
            pattern_set = pattern_set[mode] if params['split'] != 'pretrain' else pattern_set['full']

        # Get patterns for target graphs
        all_patterns = pattern_set['pattern']
        all_nid = pattern_set['nid']
        selected_patterns = all_patterns[:, graphs, :]
        selected_nid = all_nid[:, graphs, :]
        h, num_graphs, k = selected_nid.shape

        # In training, selecting a subset of patterns
        if mode == 'train':
            idx = torch.randint(0, h, (num_graphs, num_patterns))
            patterns = torch.stack([selected_patterns[idx[i], i, :] for i in range(num_graphs)], dim=1).to(device)
            nids = torch.stack([selected_nid[idx[i], i, :] for i in range(num_graphs)], dim=1).to(device)
        else:
            patterns = selected_patterns.to(device)
            nids = selected_nid.to(device)

        if graph[0].get('pe') is not None:
            node_pe_list = [graph[g].pe for g in graphs]
            max_nodes = max(pe.size(0) for pe in node_pe_list)
            dim = node_pe_list[0].size(1)
            node_pe = torch.zeros((len(node_pe_list), max_nodes, dim))
            for i, pe in enumerate(node_pe_list):
                node_pe[i, :pe.size(0), :] = pe
            node_pe = node_pe.to(device)
        else:
            node_pe = None

        if graph._data.edge_attr is not None:
            e_feat = graph._data.e_feat.to(device)
            all_eid = pattern_set['eid']
            selected_eid = all_eid[:, graphs, :]
            if mode == 'train':
                eids = torch.stack([selected_eid[idx[i], i, :] for i in range(num_graphs)], dim=1).to(device)
            else:
                eids = selected_eid.to(device)
        else:
            e_feat = None
            eids = None

        if params['dataset'] in mol_graphs:
            feat = self.atom_encoder(feat)
            if e_feat is not None:
                e_feat = self.bond_encoder(e_feat)

        pattern_feat = self.pattern_encoder.encode_graph(nids, feat, patterns, eids, e_feat, node_pe, params)

        if params['use_vq']:
            pattern_feat, _, commit_loss, _ = self.vq(pattern_feat)
        else:
            commit_loss = 0

        if params['use_cls_token']:
            pattern_feat = torch.cat([self.cls_token.repeat(1, pattern_feat.size(1), 1), pattern_feat], dim=0)
        pattern_emb = self.transformer_encode(pattern_feat)
        instance_emb = self.get_instance_emb(pattern_emb, params)

        pred = self.head(instance_emb)

        return pred, instance_emb, pattern_emb, commit_loss


class PretrainModel(BaseModel):
    def __init__(self, params):
        super(PretrainModel, self).__init__(params)

        # Initialize mask token
        self.mask_token = nn.Parameter(torch.zeros(1, self.hidden_dim),
                                       requires_grad=True if params['mask_token'] == 'learnable' else False)
        nn.init.normal_(self.mask_token, std=0.02)

        # Create online encoder and norms
        self.online_pattern_encoder = copy.deepcopy(self.pattern_encoder)
        self.online_encoder = copy.deepcopy(self.encoder)
        self.online_encoder_norm = copy.deepcopy(self.encoder_norm)

        for module in [self.online_pattern_encoder, self.online_encoder, self.online_encoder_norm]:
            for param in module.parameters():
                param.requires_grad = False

        # MAE architecture needs an additional Transformer decoder
        if params['architecture'] == 'mae':
            self.decoder_layer = nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=params["num_heads"],
                dim_feedforward=self.hidden_dim * 4,
                dropout=params["dropout"],
                norm_first=params["norm_first"]
            )
            self.decoder = nn.ModuleList([copy.deepcopy(self.decoder_layer) for _ in range(self.num_dec_layers)])
            self.decoder_norm = nn.ModuleList([nn.LayerNorm(self.hidden_dim) for _ in range(self.num_dec_layers)])
        elif params['architecture'] == 'simmim':
            # for simmim, we use linear decoder or 2-layer mlp decoder
            if params['num_dec_layers'] == 1:
                self.decoder = nn.Identity()
            elif params['num_dec_layers'] == 2:
                self.decoder = nn.Linear(self.hidden_dim, self.hidden_dim)
            else:
                raise ValueError(f"Unsupported number of decoder layers: {params['num_dec_layers']}")

        if params['objective_on'] == 'emb':
            out_dim = self.hidden_dim
        elif params['objective_on'] == 'raw_mean':
            out_dim = self.node_dim
        elif params['objective_on'] == 'raw_concat':
            out_dim = self.node_dim * (params['pattern_size'] + 1)
        else:
            raise ValueError(f"Unsupported objective on: {params['objective_on']}")
        self.linear_decoder = nn.Linear(self.hidden_dim, out_dim)

        if params['auxiliary_objective'] == 'none':
            aux_out_dim = 0
        elif params['auxiliary_objective'] == 'ap_mean':
            aux_out_dim = params['pattern_size'] + 1
        elif params['auxiliary_objective'] == 'ap_concat':
            aux_out_dim = (params['pattern_size'] + 1) * (params['pattern_size'] + 1)
        else:
            raise ValueError(f"Unsupported auxiliary objective: {params['auxiliary_objective']}")
        self.auxiliary_decoder = nn.Linear(self.hidden_dim, aux_out_dim) if aux_out_dim > 0 else None

    def online_transformer_encode(self, x):
        for layer, norm in zip(self.online_encoder, self.online_encoder_norm):
            x = layer(norm(x)) + x
        return x

    def transformer_decode(self, x):
        for layer, norm in zip(self.decoder, self.decoder_norm):
            x = layer(norm(x)) + x
        return x

    def ema_update(self, alpha=0.99):
        for online_param, param in zip(self.online_pattern_encoder.parameters(), self.pattern_encoder.parameters()):
            online_param.data.copy_(online_param.data * alpha + param.data * (1 - alpha))

        for online_layer, layer in zip(self.online_encoder, self.encoder):
            for online_param, param in zip(online_layer.parameters(), layer.parameters()):
                online_param.data.copy_(online_param.data * alpha + param.data * (1 - alpha))

        for online_norm, norm in zip(self.online_encoder_norm, self.encoder_norm):
            for online_param, param in zip(online_norm.parameters(), norm.parameters()):
                online_param.data.copy_(online_param.data * alpha + param.data * (1 - alpha))

    def pretrain_node(self, graph, nodes, params):
        device = get_device_from_model(self)

        feat = graph.x
        node_pe = graph.pe if graph.get('pe') is not None else None

        # Append a node to the feat [n, d] to facilitate pattern masking
        # Note: this only works when params['mask_pattern_ratio'] != 0
        feat = torch.cat([feat, torch.zeros(1, feat.size(1), device=feat.device)], dim=0)

        # Get patterns
        total_pattern_count = params['pre_sample_pattern_num']
        visible_pattern_count = params['num_patterns']
        masked_pattern_count = total_pattern_count - visible_pattern_count

        pattern_dict = params['pattern_set']  # TODO: this can be extracted on-the-fly
        node_patterns = pattern_dict['pattern'][:, nodes, :]
        num_patterns, num_nodes, pattern_length = node_patterns.shape

        # Shuffle patterns and use top k patterns
        shuffle_idx = torch.randperm(num_patterns)
        unshuffle_idx = torch.argsort(shuffle_idx)
        mask = torch.zeros(num_patterns, dtype=torch.bool, device=device)
        mask[shuffle_idx[visible_pattern_count:]] = True

        # Get reconstruction target
        if params['objective_on'] == 'emb':
            pattern_feat = self.online_pattern_encoder.encode_node(node_patterns, feat, node_pe, None, params)
            target = self.online_transformer_encode(pattern_feat).detach()
        elif params['objective_on'] == 'raw_mean':
            target = feat[node_patterns].mean(dim=2).to(device)
        elif params['objective_on'] == 'raw_concat':
            target = feat[node_patterns].view(num_patterns, num_nodes, -1).to(device)
        else:
            raise ValueError(f"Unsupported objective on: {params['objective_on']}")

        # Get auxiliary target
        if params['auxiliary_objective'] == 'ap_mean':
            adj = node_patterns.unsqueeze(-1) == node_patterns.unsqueeze(-2)
            adj = adj.view(-1, pattern_length, pattern_length).float()
            aux_target = adj.mean(dim=2).view(num_patterns, num_nodes, -1).to(device)
        elif params['auxiliary_objective'] == 'ap_concat':
            adj = node_patterns.unsqueeze(-1) == node_patterns.unsqueeze(-2)
            adj = adj.view(-1, pattern_length, pattern_length).float()
            aux_target = adj.view(num_patterns, num_nodes, -1).to(device)
        elif params['auxiliary_objective'] == 'none':
            pass
        else:
            raise ValueError(f"Unsupported auxiliary objective: {params['auxiliary_objective']}")

        # Apply augmentations to features and patterns
        if not params['mix_aug']:
            # Basic augmentation strategy
            feat_aug = feat
            feat_aug, _ = mask_feature(feat_aug, params['mask_feature_ratio'], mode='col', training=True)
            feat_aug, _ = mask_feature(feat_aug, params['mask_node_ratio'], mode='row', training=True)

            node_patterns_aug = node_patterns
            node_patterns_aug, _ = mask_patterns(node_patterns_aug, params['mask_pattern_ratio'], mode='mask',
                                                 training=True)
            node_patterns_aug, _ = mask_patterns(node_patterns_aug, params['replace_pattern_ratio'], mode='random',
                                                 training=True)
        else:
            # Advanced augmentation strategy
            feat_mode = random.choice(['col', 'row'])
            pattern_mode = random.choice(['mask', 'random'])

            feat_aug, _ = mask_feature(feat, params['mask_node'], mode=feat_mode, training=True)
            node_patterns_aug, _ = mask_patterns(node_patterns, params['mask_pattern'], mode=pattern_mode, training=True)

        # Get pattern features
        e_feat = None
        pattern_feat = self.pattern_encoder.encode_node(node_patterns_aug[shuffle_idx[:visible_pattern_count]], feat_aug, node_pe, e_feat, params)

        # 2. Add mask tokens for missing patterns
        if params['mask_token'] in ['learnable', 'fixed']:
            mask_tokens = self.mask_token.repeat(masked_pattern_count, pattern_feat.size(1), 1)
        elif params['mask_token'] == 'random':
            mask_tokens = torch.randn(masked_pattern_count, pattern_feat.size(1), self.hidden_dim, device=device)
        elif params['mask_token'] == 'replace':
            full_pattern_feat = self.pattern_encoder.encode_node(node_patterns_aug, feat_aug, node_pe, e_feat, params)
            mask_tokens = full_pattern_feat[torch.randperm(num_patterns)[:masked_pattern_count]].detach()
        else:
            raise ValueError(f"Unsupported mask token type: {params['mask_token']}")

        if params['architecture'] == 'mae':
            pattern_emb = self.transformer_encode(pattern_feat)
            pattern_emb_with_mask = torch.cat([pattern_emb, mask_tokens], dim=0)
            pattern_emb_with_mask = pattern_emb_with_mask[unshuffle_idx]

            recon_pattern_emb = self.transformer_decode(pattern_emb_with_mask)

        elif params['architecture'] == 'simmim':
            pattern_feat_with_mask = torch.cat([pattern_feat, mask_tokens], dim=0)

            pattern_emb = self.transformer_encode(pattern_feat_with_mask)
            pattern_emb = pattern_emb[unshuffle_idx]

            recon_pattern_emb = self.decoder(pattern_emb)

        # Calculate reconstruction loss on masked tokens only
        loss_fn = F.mse_loss if params['loss_fn'] == 'l2' else F.l1_loss

        pred = self.linear_decoder(recon_pattern_emb)  # Project to final space
        loss = loss_fn(pred[mask], target[mask])

        # Calculate auxiliary loss
        if params['auxiliary_objective'] != 'none':
            aux_pred = self.auxiliary_decoder(recon_pattern_emb)
            loss_aux = loss_fn(aux_pred[mask], aux_target[mask])
            loss = loss + loss_aux

        # Calculate consistency loss
        if params['use_consistency']:
            loss_con = loss_fn(recon_pattern_emb[~mask], target[~mask])
            loss = loss + loss_con

        return loss

    def pretrain_graph(self, graph, graphs, params):
        device = get_device_from_model(self)

        feat = graph._data.x_feat.to(device)
        # Append a node to the feat [n, d] to facilitate pattern masking
        # Note: this only works when params['mask_pattern_ratio'] != 0
        feat = torch.cat([feat, torch.zeros(1, feat.size(1), device=feat.device, dtype=feat.dtype)], dim=0)

        # Get patterns
        total_pattern_count = params['pre_sample_pattern_num']
        visible_pattern_count = params['num_patterns']
        masked_pattern_count = total_pattern_count - visible_pattern_count

        pattern_dict = params['pattern_set']  # TODO: this can be extracted on-the-fly
        graph_patterns = pattern_dict['pattern'][:, graphs, :]
        nids = pattern_dict['nid'][:, graphs, :]
        num_patterns, num_graphs, pattern_length = graph_patterns.shape
        
        if graph[0].get('pe') is not None:
            node_pe_list = [graph[g].pe for g in graphs]
            max_nodes = max(pe.size(0) for pe in node_pe_list)
            dim = node_pe_list[0].size(1)
            node_pe = torch.zeros((len(node_pe_list), max_nodes, dim))
            for i, pe in enumerate(node_pe_list):
                node_pe[i, :pe.size(0), :] = pe
            node_pe = node_pe.to(device)
        else:
            node_pe = None

        if graph._data.edge_attr is not None:
            e_feat = graph._data.e_feat.to(device)
            all_eid = pattern_dict['eid']
            selected_eid = all_eid[:, graphs, :]
            eids = selected_eid.to(device)
        else:
            e_feat = None
            eids = None

        if params['dataset'] in mol_graphs:
            feat = self.atom_encoder(feat)
            if e_feat is not None:
                e_feat = self.bond_encoder(e_feat)

        # Shuffle patterns and use top k patterns
        shuffle_idx = torch.randperm(num_patterns)
        unshuffle_idx = torch.argsort(shuffle_idx)
        mask = torch.zeros(num_patterns, dtype=torch.bool, device=device)
        mask[shuffle_idx[visible_pattern_count:]] = True

        # Get reconstruction target
        if params['objective_on'] == 'emb':
            pattern_feat = self.online_pattern_encoder.encode_graph(nids, feat, graph_patterns, eids, e_feat, node_pe, params)
            target = self.online_transformer_encode(pattern_feat).detach()
        elif params['objective_on'] == 'raw_mean':
            target = feat[graph_patterns].mean(dim=2).to(device)
        elif params['objective_on'] == 'raw_concat':
            target = feat[graph_patterns].view(num_patterns, num_graphs, -1).to(device)
        else:
            raise ValueError(f"Unsupported objective on: {params['objective_on']}")

        # Get auxiliary target
        if params['auxiliary_objective'] == 'ap_mean':
            adj = graph_patterns.unsqueeze(-1) == graph_patterns.unsqueeze(-2)
            adj = adj.view(-1, pattern_length, pattern_length).float()
            aux_target = adj.mean(dim=2).view(num_patterns, num_graphs, -1).to(device)
        elif params['auxiliary_objective'] == 'ap_concat':
            adj = graph_patterns.unsqueeze(-1) == graph_patterns.unsqueeze(-2)
            adj = adj.view(-1, pattern_length, pattern_length).float()
            aux_target = adj.view(num_patterns, num_graphs, -1).to(device)
        elif params['auxiliary_objective'] == 'none':
            pass
        else:
            raise ValueError(f"Unsupported auxiliary objective: {params['auxiliary_objective']}")

        # Apply augmentations to features and patterns
        if not params['mix_aug']:
            # Basic augmentation strategy
            feat_aug = feat
            feat_aug, _ = mask_feature(feat_aug, params['mask_feature_ratio'], mode='col', training=True)
            feat_aug, _ = mask_feature(feat_aug, params['mask_node_ratio'], mode='row', training=True)

            if e_feat is not None:  
                e_feat_aug = e_feat
                e_feat_aug, _ = mask_feature(e_feat_aug, params['mask_feature_ratio'], mode='col', training=True)
                e_feat_aug, _ = mask_feature(e_feat_aug, params['mask_node_ratio'], mode='row', training=True)

            graph_patterns_aug = graph_patterns
            graph_patterns_aug, _ = mask_patterns(graph_patterns_aug, params['mask_pattern_ratio'], mode='mask', training=True)
            graph_patterns_aug, _ = mask_patterns(graph_patterns_aug, params['replace_pattern_ratio'], mode='random', training=True)
        else:
            # Advanced augmentation strategy
            feat_mode = random.choice(['col', 'row'])
            pattern_mode = random.choice(['mask', 'random'])

            feat_aug, _ = mask_feature(feat, params['mask_node'], mode=feat_mode, training=True)
            if e_feat is not None:
                e_feat_aug, _ = mask_feature(e_feat, params['mask_node'], mode=feat_mode, training=True)
            graph_patterns_aug, _ = mask_patterns(graph_patterns, params['mask_pattern'], mode=pattern_mode, training=True)

        # Get pattern features
        if eids is not None:
            pattern_feat = self.pattern_encoder.encode_graph(nids[shuffle_idx[:visible_pattern_count]], feat_aug, graph_patterns_aug[shuffle_idx[:visible_pattern_count]], eids[shuffle_idx[:visible_pattern_count]], e_feat_aug, node_pe, params)
        else:
            pattern_feat = self.pattern_encoder.encode_graph(nids[shuffle_idx[:visible_pattern_count]], feat_aug, graph_patterns_aug[shuffle_idx[:visible_pattern_count]], None, None, node_pe, params)

        # 2. Add mask tokens for missing patterns
        if params['mask_token'] in ['learnable', 'fixed']:
            mask_tokens = self.mask_token.repeat(masked_pattern_count, pattern_feat.size(1), 1)
        elif params['mask_token'] == 'random':
            mask_tokens = torch.randn(masked_pattern_count, pattern_feat.size(1), self.hidden_dim, device=device)
        elif params['mask_token'] == 'replace':
            full_pattern_feat = self.pattern_encoder.encode_graph(nids, feat_aug, graph_patterns_aug, eids, e_feat, node_pe, params)
            mask_tokens = full_pattern_feat[torch.randperm(num_patterns)[:masked_pattern_count]].detach()
        else:
            raise ValueError(f"Unsupported mask token type: {params['mask_token']}")

        if params['architecture'] == 'mae':
            pattern_emb = self.transformer_encode(pattern_feat)
            pattern_emb_with_mask = torch.cat([pattern_emb, mask_tokens], dim=0)
            pattern_emb_with_mask = pattern_emb_with_mask[unshuffle_idx]

            recon_pattern_emb = self.transformer_decode(pattern_emb_with_mask)

        elif params['architecture'] == 'simmim':
            pattern_feat_with_mask = torch.cat([pattern_feat, mask_tokens], dim=0)

            pattern_emb = self.transformer_encode(pattern_feat_with_mask)
            pattern_emb = pattern_emb[unshuffle_idx]

            recon_pattern_emb = self.decoder(pattern_emb)

        # Calculate reconstruction loss on masked tokens only
        loss_fn = F.mse_loss if params['loss_fn'] == 'l2' else F.l1_loss

        pred = self.linear_decoder(recon_pattern_emb)  # Project to final space
        loss = loss_fn(pred[mask], target[mask])

        # Calculate auxiliary loss
        if params['auxiliary_objective'] != 'none':
            aux_pred = self.auxiliary_decoder(recon_pattern_emb)
            loss_aux = loss_fn(aux_pred[mask], aux_target[mask])
            loss = loss + loss_aux

        # Calculate consistency loss
        if params['use_consistency']:
            loss_con = loss_fn(recon_pattern_emb[~mask], target[~mask])
            loss = loss + loss_con

        return loss
