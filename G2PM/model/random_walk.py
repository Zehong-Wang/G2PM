import numpy as np

import torch
from dgl.sampling import node2vec_random_walk
from torch_cluster import random_walk

from utils.eval import *
from utils.utils import get_device_from_model, check_path


def get_patterns(graph, params, nodes=None):
    num_patterns = params["pre_sample_pattern_num"]
    bs = params["pre_sample_batch_size"]
    pattern_size = params["pattern_size"]
    p = params["p"]
    q = params["q"]

    graph = graph.to(params["device"])

    row, col = graph.edge_index
    if nodes is None:
        nodes = torch.arange(graph.num_nodes).to(params["device"])
    else:
        nodes = nodes.to(params["device"])
    num_nodes_total = nodes.size(0)
    num_batches = (num_nodes_total + bs - 1) // bs

    patterns = []
    eids = []
    for i in range(num_batches):
        cur_nodes = nodes[i * bs: (i + 1) * bs]
        num_cur_nodes = cur_nodes.size(0)
        cur_nodes = cur_nodes.repeat(num_patterns)

        cur_patterns, cur_eids = random_walk(row, col, start=cur_nodes, walk_length=pattern_size, p=p, q=q,
                                             return_edge_indices=True)

        cur_patterns = cur_patterns.view(num_patterns, num_cur_nodes, pattern_size + 1)
        cur_eids = cur_eids.view(num_patterns, num_cur_nodes, pattern_size)

        patterns.append(cur_patterns.detach().cpu().numpy())
        eids.append(cur_eids.detach().cpu().numpy())

    patterns = np.concatenate(patterns, axis=1)
    eids = np.concatenate(eids, axis=1)
    patterns = torch.from_numpy(patterns)
    eids = torch.from_numpy(eids)

    return patterns, eids


def get_patterns_for_graph(dataset, params):
    num_patterns = params["pre_sample_pattern_num"]
    bs = params["pre_sample_batch_size"]
    pattern_size = params["pattern_size"]
    p = params["p"]
    q = params["q"]

    total_patterns = []
    total_nids = []
    total_eids = []

    for graph in dataset:
        if graph.edge_index.numel() != 0:
            row, col = graph.edge_index
            nodes = torch.randint(0, graph.num_nodes, (num_patterns,))
            num_nodes_total = nodes.size(0)
            num_batches = (num_nodes_total + bs - 1) // bs

            patterns = []
            eids = []
            for i in range(num_batches):
                cur_nodes = nodes[i * bs: (i + 1) * bs]
                num_cur_nodes = cur_nodes.size(0)

                cur_patterns, cur_eids = random_walk(row, col, start=cur_nodes, walk_length=pattern_size, p=p, q=q,
                                                     return_edge_indices=True)

                cur_patterns = cur_patterns.view(num_cur_nodes, pattern_size + 1)
                cur_eids = cur_eids.view(num_cur_nodes, pattern_size)

                patterns.append(cur_patterns.detach())
                eids.append(cur_eids.detach())

            patterns = torch.cat(patterns, dim=1)
            nids = graph.x[patterns].squeeze()
            eids = torch.cat(eids, dim=1)
            if graph.edge_attr is not None:
                eids = graph.edge_attr[eids].squeeze()
        else:
            patterns = torch.zeros(num_patterns, pattern_size + 1, dtype=torch.long)
            nids = torch.zeros(num_patterns, pattern_size + 1, dtype=torch.long)
            eids = torch.zeros(num_patterns, pattern_size, dtype=torch.long)

            print('A graph with no edges is found.')

        total_patterns.append(patterns)
        total_nids.append(nids)
        total_eids.append(eids)

    patterns = torch.stack(total_patterns, dim=1)
    nids = torch.stack(total_nids, dim=1)
    eids = torch.stack(total_eids, dim=1)

    return patterns, nids, eids
