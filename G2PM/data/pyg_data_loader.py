import os.path as osp
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

from ogb.linkproppred import PygLinkPropPredDataset
from ogb.nodeproppred import PygNodePropPredDataset
from ogb.graphproppred import PygGraphPropPredDataset
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, CoraFull, Amazon, Coauthor, WikiCS, Flickr, Yelp, Reddit2, WebKB, \
    WikipediaNetwork, HeterophilousGraphDataset, Actor, LRGBDataset, GNNBenchmarkDataset, TUDataset, DeezerEurope, \
    Twitch
from .dataset.attributed_graph_dataset import AttributedGraphDataset
from .dataset.heterophily_graph_dataset import load_pokec_mat
from .dataset.transfer_learning_citation_dataset import CitationNetworkDataset
from .dataset.zinc_dataset import ZINC

import torch_geometric.transforms as T
from torch_geometric.utils import degree

from utils.utils import idx2mask, mask2idx

mol_graphs = ['esol', 'freesolv', 'lipo', 'bace', 'bbbp', 'clintox', 'hiv', 'tox21', 'toxcast', 'muv', 'pcba', 'sider',
              'zinc', 'zinc_full']


def get_split(graph, setting):
    if setting == 'da':
        train_split = 0
        val_split = 0.2
    elif setting == 'low':
        train_split = 0.1
        val_split = 0.1
    elif setting == 'median':
        train_split = 0.5
        val_split = 0.25
    elif setting == 'high':
        train_split = 0.6
        val_split = 0.2
    elif setting == 'very_high':
        train_split = 0.8
        val_split = 0.1
    elif setting == 'pretrain':
        train_split = 1.0
        val_split = 0.0
    else:
        raise ValueError("Split setting error!")

    num_nodes = graph.num_nodes
    idx = torch.randperm(num_nodes)

    train_idx = idx[:int(num_nodes * train_split)]
    val_idx = idx[int(num_nodes * train_split):int(num_nodes * (train_split + val_split))]
    test_idx = idx[int(num_nodes * (train_split + val_split)):]

    train_mask = idx2mask(train_idx, num_nodes)
    val_mask = idx2mask(val_idx, num_nodes)
    test_mask = idx2mask(test_idx, num_nodes)

    split = {'train': train_mask, 'val': val_mask, 'test': test_mask}

    return split


def get_link_split(dataset):
    edge_index = dataset.edge_index
    num_nodes = dataset.num_nodes

    num_edges = edge_index.size(1)
    train_split = 0.8
    val_split = 0.05
    test_split = 0.15

    idx = torch.randperm(num_edges)

    train_idx = idx[:int(num_edges * train_split)]
    val_idx = idx[int(num_edges * train_split):int(num_edges * (train_split + val_split))]
    test_idx = idx[int(num_edges * (train_split + val_split)):]

    train_edge_index = edge_index[:, train_idx]
    val_edge_index = edge_index[:, val_idx]
    test_edge_index = edge_index[:, test_idx]

    splits = {'train': {}, 'val': {}, 'test': {}}
    splits['train']['edge'] = train_edge_index.T
    splits['val']['edge'] = val_edge_index.T
    splits['test']['edge'] = test_edge_index.T

    # random negative sampling
    splits['val']['edge_neg'] = torch.randint(0, num_nodes, (val_edge_index.size(1), 2))
    splits['test']['edge_neg'] = torch.randint(0, num_nodes, (test_edge_index.size(1), 2))

    return splits


def get_graph_split(dataset):
    train_split = 0.8
    val_split = 0.1

    num_graphs = len(dataset)
    idx = torch.randperm(num_graphs)

    train_idx = idx[:int(num_graphs * train_split)]
    val_idx = idx[int(num_graphs * train_split):int(num_graphs * (train_split + val_split))]
    test_idx = idx[int(num_graphs * (train_split + val_split)):]

    train_mask = idx2mask(train_idx, num_graphs)
    val_mask = idx2mask(val_idx, num_graphs)
    test_mask = idx2mask(test_idx, num_graphs)

    split = {'train': train_mask, 'val': val_mask, 'test': test_mask}

    return split


def load_node_task(params):
    name = params['dataset']
    data_path = params['data_path']
    split_setting = params['split']
    repeat = params['split_repeat']

    if params['node_pe'] == 'rw':
        transform = T.Compose([T.RemoveSelfLoops(), T.ToUndirected(),
                               T.AddRandomWalkPE(params['node_pe_dim'], 'pe')])
    elif params['node_pe'] == 'lap':
        transform = T.Compose([T.RemoveSelfLoops(), T.ToUndirected(),
                               T.AddLaplacianEigenvectorPE(params['node_pe_dim'], 'pe')])
    elif params['node_pe'] == 'none':
        transform = T.Compose([T.RemoveSelfLoops(), T.ToUndirected()])
    else:
        raise ValueError("Node positional encoding error!")

    if name in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(data_path, name, transform=transform)
        graph = dataset[0]

        if split_setting == 'public':
            train_mask = graph.train_mask
            val_mask = graph.val_mask
            test_mask = graph.test_mask
            splits = [{'train': train_mask, 'val': val_mask, 'test': test_mask}] * repeat
        else:
            splits = [get_split(graph, split_setting) for _ in range(repeat)]

    elif name in ['cora_full']:
        dataset = CoraFull(root=data_path, transform=transform)
        graph = dataset[0]
        splits = [get_split(graph, split_setting) for _ in range(repeat)]

    elif name in ['acm', 'dblp']:
        dataset = CitationNetworkDataset(data_path, name, transform=transform)
        graph = dataset[0]
        splits = [get_split(graph, split_setting) for _ in range(repeat)]

    elif name in ['computers', 'photo']:
        dataset = Amazon(data_path, name, transform=transform)
        graph = dataset[0]
        splits = [get_split(graph, split_setting) for _ in range(repeat)]

    elif name in ['cs', 'physics']:
        dataset = Coauthor(data_path, name, transform=transform)
        graph = dataset[0]
        splits = [get_split(graph, split_setting) for _ in range(repeat)]

    elif name in ['wikics']:
        data_path = osp.join(data_path, 'wikics')
        dataset = WikiCS(data_path, transform=transform)
        graph = dataset[0]

        assert split_setting == 'public'

        train_mask = graph.train_mask
        val_mask = graph.val_mask
        test_mask = graph.test_mask

        splits = []
        for i in range(train_mask.shape[1]):
            splits.append({
                'train': train_mask[:, i].bool(),
                'val': val_mask[:, i].bool(),
                'test': test_mask.bool(),
            })

    elif name in ['flickr']:
        data_path = osp.join(data_path, 'flickr_large')
        dataset = Flickr(data_path, transform=transform)
        graph = dataset[0]

        assert split_setting == 'public'

        train_mask, val_mask, test_mask = graph.train_mask, graph.val_mask, graph.test_mask
        splits = [{'train': train_mask, 'val': val_mask, 'test': test_mask}] * repeat

    elif name in ['yelp']:
        data_path = osp.join(data_path, 'yelp')
        dataset = Yelp(data_path, transform=transform)
        graph = dataset[0]

        assert split_setting == 'public'

        train_mask, val_mask, test_mask = graph.train_mask, graph.val_mask, graph.test_mask
        splits = [{'train': train_mask, 'val': val_mask, 'test': test_mask}] * repeat

    elif name in ['reddit']:
        data_path = osp.join(data_path, 'reddit')
        dataset = Reddit2(data_path, transform=transform)
        graph = dataset[0]

        assert split_setting == 'public'

        train_mask, val_mask, test_mask = graph.train_mask, graph.val_mask, graph.test_mask
        splits = [{'train': train_mask, 'val': val_mask, 'test': test_mask}] * repeat

    elif name in ['arxiv', 'products']:
        data_path = osp.join(data_path, name)
        dataset = PygNodePropPredDataset(f'ogbn-{name}', root=data_path, transform=transform)
        graph = dataset[0]

        assert split_setting == 'public'

        split = dataset.get_idx_split()
        train_mask = idx2mask(split['train'], graph.num_nodes)
        val_mask = idx2mask(split['valid'], graph.num_nodes)
        test_mask = idx2mask(split['test'], graph.num_nodes)

        splits = [{'train': train_mask, 'val': val_mask, 'test': test_mask}] * repeat

        return graph, splits

    elif name in ['proteins']:
        # get node features
        data_path = osp.join(data_path, name)
        dataset = PygNodePropPredDataset(f'ogbn-proteins', root=data_path, transform=transform)
        graph = dataset[0]

        preprocess = T.ToSparseTensor(attr='edge_attr')
        graph.x = preprocess(graph).adj_t.mean(dim=1)
        graph.edge_attr = None

        assert split_setting == 'public'

        split = dataset.get_idx_split()
        train_mask = idx2mask(split['train'], graph.num_nodes)
        val_mask = idx2mask(split['valid'], graph.num_nodes)
        test_mask = idx2mask(split['test'], graph.num_nodes)

        splits = [{'train': train_mask, 'val': val_mask, 'test': test_mask}] * repeat

        return graph, splits

    elif name in ['pokec']:
        dataset = load_pokec_mat(data_path)

        x = dataset.graph['node_feat']
        edge_index = dataset.graph['edge_index']
        y = dataset.label

        graph = Data(x=x, edge_index=edge_index, y=y)
        graph = transform(graph)

        assert split_setting == 'public'

        split = dataset.get_idx_split()
        train_mask = idx2mask(split['train'], dataset.graph['num_nodes'])
        val_mask = idx2mask(split['valid'], dataset.graph['num_nodes'])
        test_mask = idx2mask(split['test'], dataset.graph['num_nodes'])

        splits = [{'train': train_mask, 'val': val_mask, 'test': test_mask}] * repeat

        return graph, splits

    elif name in ['blog', 'flickr_small', 'PPI']:
        name_map = {'blog': 'BlogCatalog', 'flickr_small': 'Flickr'}
        name = name_map[name]
        dataset = AttributedGraphDataset(data_path, name, transform=transform)
        graph = dataset[0]

        splits = [get_split(graph, split_setting) for _ in range(repeat)]

        return graph, splits

    elif name in ['cornell', 'wisconsin', 'texas']:
        dataset = WebKB(data_path, name, transform=transform)
        graph = dataset[0]

        if split_setting == 'public':
            train_mask = graph.train_mask
            val_mask = graph.val_mask
            test_mask = graph.test_mask

            splits = []
            for i in range(train_mask.shape[1]):
                splits.append({
                    'train': train_mask[:, i].bool(),
                    'val': val_mask[:, i].bool(),
                    'test': test_mask[:, i].bool(),
                })
        else:
            splits = [get_split(graph, split_setting) for _ in range(repeat)]

    elif name in ['chameleon', 'squirrel']:
        dataset = WikipediaNetwork(data_path, name, transform=transform, geom_gcn_preprocess=True)
        graph = dataset[0]

        if split_setting == 'public':
            train_mask = graph.train_mask
            val_mask = graph.val_mask
            test_mask = graph.test_mask

            splits = []
            for i in range(train_mask.shape[1]):
                splits.append({
                    'train': train_mask[:, i].bool(),
                    'val': val_mask[:, i].bool(),
                    'test': test_mask[:, i].bool(),
                })
        else:
            splits = [get_split(graph, split_setting) for _ in range(repeat)]

        return graph, splits

    elif name in ['DE', 'EN', 'ES', 'FR', 'PT', 'RU']:
        dataset = Twitch(data_path, name, transform=transform)
        graph = dataset[0]

        splits = [get_split(graph, split_setting) for _ in range(repeat)]

    elif name in ['deezer']:
        dataset = DeezerEurope(data_path, transform=transform)
        graph = dataset[0]

        splits = [get_split(graph, split_setting) for _ in range(repeat)]

    elif name in ['roman', 'ratings', 'minesweeper', 'tolokers', 'questions']:
        name_map = {'roman': "roman_empire", "ratings": "amazon-ratings", "minesweeper": "minesweeper",
                    "tolokers": "tolokers", "questions": "questions"}
        name = name_map[name]

        dataset = HeterophilousGraphDataset(data_path, name, transform=transform)
        graph = dataset[0]

        if split_setting == 'public':
            train_mask = graph.train_mask
            val_mask = graph.val_mask
            test_mask = graph.test_mask

            splits = []
            for i in range(train_mask.shape[1]):
                splits.append({
                    'train': train_mask[:, i].bool(),
                    'val': val_mask[:, i].bool(),
                    'test': test_mask[:, i].bool(),
                })

        else:
            splits = [get_split(graph, split_setting) for _ in range(repeat)]

    elif name in ['actor']:
        data_path = osp.join(data_path, 'actor')
        dataset = Actor(data_path, transform=transform)
        graph = dataset[0]

        if split_setting == 'public':
            train_mask = graph.train_mask
            val_mask = graph.val_mask
            test_mask = graph.test_mask

            splits = []
            for i in range(train_mask.shape[1]):
                splits.append({
                    'train': train_mask[:, i].bool(),
                    'val': val_mask[:, i].bool(),
                    'test': test_mask[:, i].bool(),
                })

        else:
            splits = [get_split(graph, split_setting) for _ in range(repeat)]

    elif name in ['pascalvoc-sp', 'coco-sp']:
        train_set = LRGBDataset(data_path, name, split='train', transform=transform)
        val_set = LRGBDataset(data_path, name, split='val', transform=transform)
        test_set = LRGBDataset(data_path, name, split='test', transform=transform)

        # TODO: Append positional encoding to node features

        return {'train': train_set, 'val': val_set, 'test': test_set}, None

    elif name in ['PATTERN', 'CLUSTER']:
        train_set = GNNBenchmarkDataset(data_path, name, split='train', transform=transform)
        val_set = GNNBenchmarkDataset(data_path, name, split='val', transform=transform)
        test_set = GNNBenchmarkDataset(data_path, name, split='test', transform=transform)

        # TODO: Append positional encoding to node features

        return {'train': train_set, 'val': val_set, 'test': test_set}, None

    else:
        raise ValueError("Dataset name error!")

    return graph, splits


def load_link_task(params):
    name = params['dataset']
    data_path = params['data_path']

    if params['node_pe'] == 'rw':
        transform = T.Compose([T.RemoveSelfLoops(), T.ToUndirected(),
                               T.AddRandomWalkPE(params['node_pe_dim'], 'pe')])
    elif params['node_pe'] == 'lap':
        transform = T.Compose([T.RemoveSelfLoops(), T.ToUndirected(),
                               T.AddLaplacianEigenvectorPE(params['node_pe_dim'], 'pe')])
    elif params['node_pe'] == 'none':
        transform = T.Compose([T.RemoveSelfLoops(), T.ToUndirected()])
    else:
        raise ValueError("Node positional encoding error!")

    if name in ['pcqm-contact']:
        train_set = LRGBDataset(data_path, name, split='train')
        val_set = LRGBDataset(data_path, name, split='val')
        test_set = LRGBDataset(data_path, name, split='test')

        return {'train': train_set, 'val': val_set, 'test': test_set}, None

    elif name in ['link-collab', 'link-ppa', 'link-ddi']:
        if params['node_pe'] == 'rw':
            transform = T.Compose([T.RemoveSelfLoops(), T.AddRandomWalkPE(params['node_pe_dim'], 'pe')])
        elif params['node_pe'] == 'lap':
            transform = T.Compose([T.RemoveSelfLoops(), T.AddLaplacianEigenvectorPE(params['node_pe_dim'], 'pe')])
        elif params['node_pe'] == 'none':
            transform = T.Compose([T.RemoveSelfLoops()])
        else:
            raise ValueError("Node positional encoding error!")

        name_maps = {'link-collab': 'ogbl-collab', 'link-ppa': 'ogbl-ppa', 'link-ddi': 'ogbl-ddi'}
        name = name_maps[name]

        dataset = PygLinkPropPredDataset(name, root=data_path, transform=transform)
        data = dataset[0]
        split_edge = dataset.get_edge_split()

        if name == 'ogbl-ddi':
            num_nodes = data.num_nodes
            data.x = torch.randn(num_nodes, params['hidden_dim'])

        if name == 'ogbl-ppa':
            # convert data.x to float
            data.x = data.x.float()

        split_edge['val'] = split_edge['valid']
        del split_edge['valid']
        splits = [split_edge] * params['split_repeat']

        return data, splits

    elif name in ['link-cora', 'link-pubmed', 'link-citeseer']:
        name_maps = {'link-cora': 'cora', 'link-pubmed': 'pubmed', 'link-citeseer': 'citeseer'}
        name = name_maps[name]

        dataset = Planetoid(data_path, name, transform=transform)
        graph = dataset[0]
        splits = [get_link_split(graph) for _ in range(params['split_repeat'])]

        return graph, splits

    elif name in ['link-photo', 'link-computers']:
        name_maps = {'link-photo': 'photo', 'link-computers': 'computers'}
        name = name_maps[name]

        dataset = Amazon(data_path, name, transform=transform)
        graph = dataset[0]
        splits = [get_link_split(graph) for _ in range(params['split_repeat'])]

        return graph, splits


def load_graph_task(params):
    name = params['dataset']
    data_path = params['data_path']
    split_setting = params['split']

    assert split_setting == 'public'

    if params['node_pe'] == 'rw':
        transform = T.Compose([T.AddRandomWalkPE(params['node_pe_dim'], 'pe')])
    elif params['node_pe'] == 'lap':
        transform = T.Compose([T.AddLaplacianEigenvectorPE(params['node_pe_dim'], 'pe')])
    elif params['node_pe'] == 'none':
        transform = None
    else:
        raise ValueError("Node positional encoding error!")

    if name in ['zinc', 'zinc_full']:
        subset = name == 'zinc'
        data_path = osp.join(data_path, name)
        train_set = ZINC(data_path, subset=subset, split='train', pre_transform=transform)
        val_set = ZINC(data_path, subset=subset, split='val', pre_transform=transform)
        test_set = ZINC(data_path, subset=subset, split='test', pre_transform=transform)
        full_set = ZINC(data_path, subset=subset, split='full', pre_transform=transform)

        train_set._data.edge_attr = train_set._data.edge_attr.unsqueeze(1) - 1
        val_set._data.edge_attr = val_set._data.edge_attr.unsqueeze(1) - 1
        test_set._data.edge_attr = test_set._data.edge_attr.unsqueeze(1) - 1
        full_set._data.edge_attr = full_set._data.edge_attr.unsqueeze(1) - 1

        train_set._data.y = train_set._data.y.unsqueeze(1)
        val_set._data.y = val_set._data.y.unsqueeze(1)
        test_set._data.y = test_set._data.y.unsqueeze(1)
        full_set._data.y = full_set._data.y.unsqueeze(1)

        x = torch.concatenate([train_set.x, val_set.x, test_set.x], dim=0)
        edge_attr = torch.concatenate([train_set.edge_attr, val_set.edge_attr, test_set.edge_attr], dim=0)

        unique_x, _ = np.unique(x, axis=0, return_inverse=True)
        unique_edge_attr, _ = np.unique(edge_attr, axis=0, return_inverse=True)

        unique_x = torch.tensor(unique_x, dtype=torch.long)
        unique_edge_attr = torch.tensor(unique_edge_attr, dtype=torch.long)

        x_feat = unique_x
        e_feat = unique_edge_attr

        train_set._data.x_feat = x_feat
        train_set._data.e_feat = e_feat

        val_set._data.x_feat = x_feat
        val_set._data.e_feat = e_feat

        test_set._data.x_feat = x_feat
        test_set._data.e_feat = e_feat

        full_set._data.x_feat = x_feat
        full_set._data.e_feat = e_feat

        num_train = train_set._data.y.shape[0]
        num_val = val_set._data.y.shape[0]
        num_test = test_set._data.y.shape[0]
        num_full = full_set._data.y.shape[0]

        train_mask = torch.zeros(num_full, dtype=torch.bool)
        train_mask[:num_train] = True
        val_mask = torch.zeros(num_full, dtype=torch.bool)
        val_mask[num_train:num_train + num_val] = True
        test_mask = torch.zeros(num_full, dtype=torch.bool)
        test_mask[num_train + num_val:num_train + num_val + num_test] = True

        splits = [{'train': train_mask, 'val': val_mask, 'test': test_mask}] * params['split_repeat']

        return full_set, splits

    # OGB Molecule datasets
    elif name in ['esol', 'freesolv', 'lipo', 'muv', 'bace', 'bbbp', 'tox21', 'toxcast', 'sider', 'clintox', 'hiv', 'pcba']:
        name_map = {
            'esol': 'ogbg-molesol', 'freesolv': 'ogbg-molfreesolv', 'lipo': 'ogbg-mollipo', 'muv': 'ogbg-molmuv',
            'pcba': 'ogbg-molpcba', 'bace': 'ogbg-molbace', 'bbbp': 'ogbg-molbbbp', 'tox21': 'ogbg-moltox21',
            'toxcast': 'ogbg-moltoxcast', 'sider': 'ogbg-molsider', 'clintox': 'ogbg-molclintox', 'hiv': 'ogbg-molhiv'
        }
        name = name_map[name]

        dataset = PygGraphPropPredDataset(name, root=data_path, pre_transform=transform)

        unique_x, x_idx = np.unique(dataset.data.x, axis=0, return_inverse=True)
        unique_edge_attr, edge_attr_idx = np.unique(dataset.data.edge_attr, axis=0, return_inverse=True)

        unique_x = torch.tensor(unique_x, dtype=torch.long)
        unique_edge_attr = torch.tensor(unique_edge_attr, dtype=torch.long)
        x_idx = torch.tensor(x_idx, dtype=torch.long).unsqueeze(-1)
        edge_attr_idx = torch.tensor(edge_attr_idx, dtype=torch.long).unsqueeze(-1)

        x_feat = unique_x
        e_feat = unique_edge_attr

        dataset._data.x_feat = x_feat
        dataset._data.e_feat = e_feat
        dataset._data.x = x_idx
        dataset._data.edge_attr = edge_attr_idx

        if name in ['ogbg-molesol', 'ogbg-molfreesolv', 'ogbg-mollipo']:
            mean, std = dataset.data.y.mean(), dataset.data.y.std()
            dataset._data.y = (dataset.data.y - mean) / std

        if split_setting == 'public':
            split_idx = dataset.get_idx_split()
            train_set = dataset[split_idx['train']]
            val_set = dataset[split_idx['valid']]
            test_set = dataset[split_idx['test']]

            train_mask = split_idx['train']
            val_mask = split_idx['valid']
            test_mask = split_idx['test']

            # return {'train': train_set, 'val': val_set, 'test': test_set}, None
            return dataset, [{'train': train_mask, 'val': val_mask, 'test': test_mask}] * params['split_repeat']

        else:
            splits = [get_graph_split(dataset)] * params['split_repeat']
            return dataset, splits

    elif name in ['mutag', 'nci1', 'dd', 'proteins', 'enzymes']:
        assert split_setting == 'public'

        name_map = {'mutag': 'MUTAG', 'nci1': 'NCI1', 'dd': 'DD', 'proteins': 'PROTEINS', 'enzymes': 'ENZYMES'}
        name = name_map[name]

        dataset = TUDataset(root=data_path, name=name, use_node_attr=True, use_edge_attr=True, transform=transform)
        unique_x, x_idx = np.unique(dataset._data.x, axis=0, return_inverse=True)
        unique_x = torch.tensor(unique_x, dtype=torch.long)
        x_idx = torch.tensor(x_idx, dtype=torch.long).unsqueeze(-1)
        x_feat = unique_x.float()
        dataset._data.x_feat = x_feat
        dataset._data.x = x_idx

        if dataset._data.edge_attr is not None:
            unique_edge_attr, edge_attr_idx = np.unique(dataset._data.edge_attr, axis=0, return_inverse=True)
            unique_edge_attr = torch.tensor(unique_edge_attr, dtype=torch.long)
            edge_attr_idx = torch.tensor(edge_attr_idx, dtype=torch.long).unsqueeze(-1)
            e_feat = unique_edge_attr
            dataset._data.e_feat = e_feat.float()
            dataset._data.edge_attr = edge_attr_idx

        splits = [get_graph_split(dataset)] * params['split_repeat']

        return dataset, splits

    elif name in ['collab', 'imdb-b', 'imdb-m', 'reddit-b', 'reddit-m5k', 'reddit-m12k']:
        assert split_setting == 'public'
        name_map = {'collab': 'COLLAB', 'imdb-b': 'IMDB-BINARY', 'imdb-m': 'IMDB-MULTI', 'reddit-b': 'REDDIT-BINARY',
                    'reddit-m5k': 'REDDIT-MULTI-5K', 'reddit-m12k': 'REDDIT-MULTI-12K'}
        name = name_map[name]
        MAX_DEG = 400

        # Pre-transformation version to prevent OOM
        if params['node_pe'] == 'rw':
            pre_transform = T.Compose([T.Constant(1), T.AddRandomWalkPE(params['node_pe_dim'], 'pe')])
        elif params['node_pe'] == 'lap':
            pre_transform = T.Compose([T.Constant(1), T.AddLaplacianEigenvectorPE(params['node_pe_dim'], 'pe')])
        elif params['node_pe'] == 'none':
            pre_transform = T.Constant(1)
        else:
            raise ValueError("Node positional encoding error!")

        dataset = TUDataset(root=data_path, name=name, use_node_attr=True, use_edge_attr=True,
                            pre_transform=pre_transform, force_reload=True)

        degree_list = []
        for g in dataset:
            degrees = degree(g.edge_index[0], g.num_nodes, dtype=torch.long)
            degrees[degrees > MAX_DEG] = MAX_DEG
            degree_list.append(degrees)
        degrees = torch.cat(degree_list, dim=0)
        degrees[degrees > MAX_DEG] = MAX_DEG

        dataset.data.x = degrees.unsqueeze(1).long()
        num_feat = dataset.x.max().item() + 1
        dataset._data.x_feat = F.one_hot(torch.arange(num_feat), num_classes=num_feat).float()

        splits = [get_graph_split(dataset)] * params['split_repeat']

        return dataset, splits

    elif name in ['ppa', 'code2']:
        name_map = {'ppa': 'ogbg-ppa', 'code2': 'ogbg-code2'}
        name = name_map[name]

        dataset = PygGraphPropPredDataset(name, root=data_path, transform=transform)

        split_idx = dataset.get_idx_split()
        train_set = dataset[split_idx['train']]
        val_set = dataset[split_idx['valid']]
        test_set = dataset[split_idx['test']]

        return {'train': train_set, 'val': val_set, 'test': test_set}, None

    elif name in ['func', 'struct']:
        name_map = {'func': 'peptides-func', 'struct': 'peptides-struct'}
        name = name_map[name]

        train_set = LRGBDataset(data_path, name, split='train', transform=transform)
        val_set = LRGBDataset(data_path, name, split='val', transform=transform)
        test_set = LRGBDataset(data_path, name, split='test', transform=transform)

        datasets = {'train': train_set, 'val': val_set, 'test': test_set}

        for split in ['train', 'val', 'test']:
            dataset = datasets[split]

            unique_x, x_idx = np.unique(dataset._data.x, axis=0, return_inverse=True)
            unique_x = torch.tensor(unique_x, dtype=torch.long)
            x_idx = torch.tensor(x_idx, dtype=torch.long).unsqueeze(-1)
            x_feat = unique_x
            dataset._data.x_feat = x_feat
            dataset._data.x = x_idx

            if dataset._data.edge_attr is not None:
                unique_edge_attr, edge_attr_idx = np.unique(dataset._data.edge_attr, axis=0, return_inverse=True)
                unique_edge_attr = torch.tensor(unique_edge_attr, dtype=torch.long)
                edge_attr_idx = torch.tensor(edge_attr_idx, dtype=torch.long).unsqueeze(-1)
                e_feat = unique_edge_attr
                dataset._data.e_feat = e_feat
                dataset._data.edge_attr = edge_attr_idx

            datasets[split] = dataset

        return datasets, None

    elif name in ['MNIST', 'CIFAR10', 'TSP']:
        train_set = GNNBenchmarkDataset(data_path, name, split='train', transform=transform)
        val_set = GNNBenchmarkDataset(data_path, name, split='val', transform=transform)
        test_set = GNNBenchmarkDataset(data_path, name, split='test', transform=transform)

        return {'train': train_set, 'val': val_set, 'test': test_set}, None


def load_data(params):
    task = params['task']

    if task == 'node':
        return load_node_task(params)
    elif task == 'link':
        return load_link_task(params)
    elif task == 'graph':
        return load_graph_task(params)
    else:
        raise NotImplementedError('The function is not implemented yet!')
