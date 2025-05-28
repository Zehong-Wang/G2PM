import yaml
import os.path as osp
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from ogb.linkproppred import *
from ogb.nodeproppred import *

from model.random_walk import get_patterns

from utils.eval import evaluate
from utils.utils import get_device_from_model, seed_everything, check_path, get_num_params, to_millions, mask2idx


# def multitask_cross_entropy(y_pred, y):
#     criterion = nn.BCEWithLogitsLoss(reduction="none")
#
#     loss = 0.0
#     for idx in range(y.shape[1]):
#         cur_y = y[:, idx]
#         cur_pred = y_pred[:, idx]
#         task_loss = criterion(cur_pred.double(), cur_y)
#         loss += torch.mean(task_loss)
#
#     return loss / y.shape[1]


def preprocess_link(graph, params):
    pre_sample_pattern_num = params['pre_sample_pattern_num']
    pattern_size = params['pattern_size']
    p = params['p']
    q = params['q']

    if isinstance(graph, dict):
        pattern_dir = osp.join(params['pattern_path'], params['dataset'])
        pattern_dict = {}
        for key, g_set in graph.items():
            cur_dir = osp.join(pattern_dir, key, f"{pre_sample_pattern_num}_{pattern_size}_{p}_{q}")
            check_path(cur_dir)
            pattern_dict[key] = {}

            for i, g in enumerate(g_set):
                pattern_path = osp.join(cur_dir, f"ptn_{i}.pt")
                eid_path = osp.join(cur_dir, f"eid_{i}.pt")
                if osp.exists(pattern_path) and osp.exists(eid_path):
                    patterns = torch.load(pattern_path, map_location=torch.device('cpu'))
                    eids = torch.load(eid_path, map_location=torch.device('cpu'))
                else:
                    patterns, eids = get_patterns(g, params)
                    torch.save(patterns, pattern_path)
                    torch.save(eids, eid_path)
                pattern_dict[key][i] = {'pattern': patterns, 'eid': eids}
        return pattern_dict
    else:
        pattern_dir = osp.join(params['pattern_path'], params['dataset'])
        check_path(pattern_dir)

        pattern_path = osp.join(pattern_dir, f"ptn_{pre_sample_pattern_num}_{pattern_size}_{p}_{q}.pt")
        eid_path = osp.join(pattern_dir, f"eid_{pre_sample_pattern_num}_{pattern_size}_{p}_{q}.pt")
        if osp.exists(pattern_path) and osp.exists(eid_path):
            patterns = torch.load(pattern_path, map_location=torch.device('cpu'))
            eids = torch.load(eid_path, map_location=torch.device('cpu'))
            print('Done loading patterns from cache.')
        else:
            patterns, eids = get_patterns(graph, params)
            torch.save(patterns, pattern_path)
            torch.save(eids, eid_path)

        return {'pattern': patterns, 'eid': eids}


def train_link(graph, model, optimizer, split, scheduler=None, params=None):
    if params['inference']:
        return {'train': 0, 'val': 0, 'test': 0}

    model.train()
    device = get_device_from_model(model)
    bs = params['batch_size']

    total_loss, total_val_loss, total_test_loss = 0, 0, 0
    pos_train_edge = split['train']['edge']

    train_num_edges = pos_train_edge.size(0)
    train_num_batches = (train_num_edges + bs - 1) // bs
    train_perm = torch.randperm(train_num_edges)

    for i in range(train_num_batches):
        cur_pos_edges = pos_train_edge[train_perm[i * bs: (i + 1) * bs]]
        cur_neg_edges = torch.randint(0, graph.num_nodes, cur_pos_edges.size(), dtype=torch.long,
                                      device=cur_pos_edges.device)

        pos_pred, _, _, _ = model(graph, cur_pos_edges, params, mode='train')
        neg_pred, _, _, _ = model(graph, cur_neg_edges, params, mode='train')

        pos_loss = -torch.log(pos_pred + 1e-15).mean()
        neg_loss = -torch.log(1 - neg_pred + 1e-15).mean()
        loss = pos_loss + neg_loss

        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        if params['grad_clip'] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), params['grad_clip'])
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

    total_loss /= train_num_batches
    total_val_loss /= train_num_batches
    total_test_loss /= train_num_batches

    return {'train': total_loss, 'val': total_val_loss, 'test': total_test_loss}


def eval_link(graph, model, split, params):
    model.eval()
    device = get_device_from_model(model)

    bs = params['batch_size']
    results = {'train': 0, 'metric': params['metric']}

    with torch.no_grad():
        # Acquire node embeddings
        tmp_params = params.copy()
        tmp_params['task'] = 'node'

        nodes = torch.arange(graph.num_nodes)
        num_batches = (graph.num_nodes + bs - 1) // bs
        node_embs = []
        for i in range(num_batches):
            cur_nodes = nodes[i * bs: (i + 1) * bs]
            _, emb, _, _ = model(graph, cur_nodes, tmp_params, mode='eval')
            node_embs.append(emb.detach())
        node_embs = torch.cat(node_embs, dim=0)

        for key in ['val', 'test']:
            pos_edge = split[key]['edge']
            neg_edge = split[key]['edge_neg']

            num_pos_batches = (pos_edge.size(0) + bs - 1) // bs
            num_neg_batches = (neg_edge.size(0) + bs - 1) // bs
            pos_pred_list, neg_pred_list = [], []

            for i in range(num_pos_batches):
                edge = pos_edge[i * bs: (i + 1) * bs]
                edge_emb = node_embs[edge[:, 0]] * node_embs[edge[:, 1]]
                pred = model.head(edge_emb).squeeze()
                pos_pred_list.append(pred.detach().cpu())
            pos_pred = torch.cat(pos_pred_list, dim=0)

            for i in range(num_neg_batches):
                edge = neg_edge[i * bs: (i + 1) * bs]
                edge_emb = node_embs[edge[:, 0]] * node_embs[edge[:, 1]]
                pred = model.head(edge_emb).squeeze()
                neg_pred_list.append(pred.detach().cpu())
            neg_pred = torch.cat(neg_pred_list, dim=0)

            results[key] = evaluate(pos_pred, neg_pred, params=params)
    return results
