from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
import scipy
import scipy.io
from sklearn.preprocessing import label_binarize
import torch_geometric.transforms as T

# from data_utils import rand_train_test_idx, even_quantile_labels, to_sparse_tensor, dataset_drive_url, class_rand_splits

from torch_geometric.datasets import Planetoid, Amazon, Coauthor
from torch_geometric.transforms import NormalizeFeatures
from os import path

from torch_sparse import SparseTensor
from google_drive_downloader import GoogleDriveDownloader as gdd

import networkx as nx
import scipy.sparse as sp

from ogb.nodeproppred import NodePropPredDataset, PygNodePropPredDataset
import os

from torch_geometric.utils import subgraph, k_hop_subgraph, to_undirected
import pickle as pkl

# Helper functions


dataset_drive_url = {
    'snap-patents': '1ldh23TSY1PwXia6dU0MYcpyEgX-w3Hia',
    'pokec': '1dNs5E7BrWJbgcHeQ_zuy5Ozp2tRCWG0y',
    'yelp-chi': '1fAXtTVQS4CfEk4asqrFw9EPmlUPGbGtJ',
}

splits_drive_url = {
    'snap-patents': '12xbBRqd8mtG_XkNLH8dRRNZJvVM4Pw-N',
    'pokec': '1ZhpAiyTNc0cE_hhgyiqxnkKREHK7MK-_',
}


def rand_train_test_idx(label, train_prop=.5, valid_prop=.25, ignore_negative=True):
    """ randomly splits label into train/valid/test splits """
    if ignore_negative:
        labeled_nodes = torch.where(label != -1)[0]
    else:
        labeled_nodes = label

    n = labeled_nodes.shape[0]
    train_num = int(n * train_prop)
    valid_num = int(n * valid_prop)

    perm = torch.as_tensor(np.random.permutation(n))

    train_indices = perm[:train_num]
    val_indices = perm[train_num:train_num + valid_num]
    test_indices = perm[train_num + valid_num:]

    if not ignore_negative:
        return train_indices, val_indices, test_indices

    train_idx = labeled_nodes[train_indices]
    valid_idx = labeled_nodes[val_indices]
    test_idx = labeled_nodes[test_indices]

    return train_idx, valid_idx, test_idx


def load_fixed_splits(data_dir, dataset, name, protocol):
    splits_lst = []
    if name in ['cora', 'citeseer', 'pubmed'] and protocol == 'semi':
        splits = {}
        splits['train'] = torch.as_tensor(dataset.train_idx)
        splits['valid'] = torch.as_tensor(dataset.valid_idx)
        splits['test'] = torch.as_tensor(dataset.test_idx)
        splits_lst.append(splits)
    elif name in ['cora', 'citeseer', 'pubmed', 'chameleon', 'squirrel', 'film', 'cornell', 'texas', 'wisconsin']:
        for i in range(10):
            splits_file_path = '{}/geom-gcn/splits/{}'.format(data_dir, name) + '_split_0.6_0.2_' + str(i) + '.npz'
            splits = {}
            with np.load(splits_file_path) as splits_file:
                splits['train'] = torch.BoolTensor(splits_file['train_mask'])
                splits['valid'] = torch.BoolTensor(splits_file['val_mask'])
                splits['test'] = torch.BoolTensor(splits_file['test_mask'])
            splits_lst.append(splits)
    else:
        raise NotImplementedError

    return splits_lst


def class_rand_splits(label, label_num_per_class, valid_num=500, test_num=1000):
    train_idx, non_train_idx = [], []
    idx = torch.arange(label.shape[0])
    class_list = label.squeeze().unique()
    for i in range(class_list.shape[0]):
        c_i = class_list[i]
        idx_i = idx[label.squeeze() == c_i]
        n_i = idx_i.shape[0]
        rand_idx = idx_i[torch.randperm(n_i)]
        train_idx += rand_idx[:label_num_per_class].tolist()
        non_train_idx += rand_idx[label_num_per_class:].tolist()
    train_idx = torch.as_tensor(train_idx)
    non_train_idx = torch.as_tensor(non_train_idx)
    non_train_idx = non_train_idx[torch.randperm(non_train_idx.shape[0])]
    valid_idx, test_idx = non_train_idx[:valid_num], non_train_idx[valid_num:valid_num + test_num]

    return train_idx, valid_idx, test_idx


def even_quantile_labels(vals, nclasses, verbose=True):
    """ partitions vals into nclasses by a quantile based split,
    where the first class is less than the 1/nclasses quantile,
    second class is less than the 2/nclasses quantile, and so on

    vals is np array
    returns an np array of int class labels
    """
    label = -1 * np.ones(vals.shape[0], dtype=np.int)
    interval_lst = []
    lower = -np.inf
    for k in range(nclasses - 1):
        upper = np.quantile(vals, (k + 1) / nclasses)
        interval_lst.append((lower, upper))
        inds = (vals >= lower) * (vals < upper)
        label[inds] = k
        lower = upper
    label[vals >= lower] = nclasses - 1
    interval_lst.append((lower, np.inf))
    if verbose:
        print('Class Label Intervals:')
        for class_idx, interval in enumerate(interval_lst):
            print(f'Class {class_idx}: [{interval[0]}, {interval[1]})]')
    return label


class NCDataset(object):
    def __init__(self, name):
        """
        based off of ogb NodePropPredDataset
        https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/dataset.py
        Gives torch tensors instead of numpy arrays
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder
            - meta_dict: dictionary that stores all the meta-information about data. Default is None,
                    but when something is passed, it uses its information. Useful for debugging for external contributers.

        Usage after construction:

        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph, label = dataset[0]

        Where the graph is a dictionary of the following form:
        dataset.graph = {'edge_index': edge_index,
                         'edge_feat': None,
                         'node_feat': node_feat,
                         'num_nodes': num_nodes}
        For additional documentation, see OGB Library-Agnostic Loader https://ogb.stanford.edu/docs/nodeprop/

        """

        self.name = name  # original name, e.g., ogbn-proteins
        self.graph = {}
        self.label = None

    def get_idx_split(self, split_type='random', train_prop=.5, valid_prop=.25, label_num_per_class=20):
        """
        train_prop: The proportion of dataset for train split. Between 0 and 1.
        valid_prop: The proportion of dataset for validation split. Between 0 and 1.
        """

        if split_type == 'random':
            ignore_negative = False if self.name == 'ogbn-proteins' else True
            train_idx, valid_idx, test_idx = rand_train_test_idx(
                self.label, train_prop=train_prop, valid_prop=valid_prop, ignore_negative=ignore_negative)
            split_idx = {'train': train_idx,
                         'valid': valid_idx,
                         'test': test_idx}
        elif split_type == 'class':
            train_idx, valid_idx, test_idx = class_rand_splits(self.label, label_num_per_class=label_num_per_class)
            split_idx = {'train': train_idx,
                         'valid': valid_idx,
                         'test': test_idx}
        return split_idx

    def __getitem__(self, idx):
        assert idx == 0, 'This dataset has only one graph'
        return self.graph, self.label

    def __len__(self):
        return 1

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, len(self))


def load_dataset(data_dir, dataname, sub_dataname=''):
    """ Loader for NCDataset
        Returns NCDataset
    """
    print(dataname)
    if dataname == 'pokec':
        dataset = load_pokec_mat(data_dir)
    elif dataname == 'snap-patents':
        dataset = load_snap_patents_mat(data_dir)
    elif dataname == 'amazon2m':
        dataset = load_amazon2m_dataset(data_dir)
    elif dataname == 'ogbn-papers100M':
        dataset = load_papers100M(data_dir)
    elif dataname == 'ogbn-papers100M-sub':
        dataset = papers100M_sub(data_dir)
    else:
        raise ValueError('Invalid dataname')
    return dataset


def load_amazon2m_dataset(data_dir):
    ogb_dataset = NodePropPredDataset(name='ogbn-products', root=f'{data_dir}/ogb')
    dataset = NCDataset('amazon2m')
    dataset.graph = ogb_dataset.graph
    dataset.graph['edge_index'] = torch.as_tensor(dataset.graph['edge_index'])
    dataset.graph['node_feat'] = torch.as_tensor(dataset.graph['node_feat'])
    dataset.label = torch.as_tensor(ogb_dataset.labels).reshape(-1, 1)

    def load_fixed_splits(train_prop=0.5, val_prop=0.25):
        dir = f'{data_dir}ogb/ogbn_products/split/random_0.5_0.25'
        tensor_split_idx = {}
        if os.path.exists(dir):
            tensor_split_idx['train'] = torch.as_tensor(np.loadtxt(dir + '/amazon2m_train.txt'), dtype=torch.long)
            tensor_split_idx['valid'] = torch.as_tensor(np.loadtxt(dir + '/amazon2m_valid.txt'), dtype=torch.long)
            tensor_split_idx['test'] = torch.as_tensor(np.loadtxt(dir + '/amazon2m_test.txt'), dtype=torch.long)
        else:
            os.makedirs(dir)
            tensor_split_idx['train'], tensor_split_idx['valid'], tensor_split_idx['test'] \
                = rand_train_test_idx(dataset.label, train_prop=train_prop, valid_prop=val_prop)
            np.savetxt(dir + '/amazon2m_train.txt', tensor_split_idx['train'], fmt='%d')
            np.savetxt(dir + '/amazon2m_valid.txt', tensor_split_idx['valid'], fmt='%d')
            np.savetxt(dir + '/amazon2m_test.txt', tensor_split_idx['test'], fmt='%d')
        return tensor_split_idx

    dataset.load_fixed_splits = load_fixed_splits
    return dataset


def load_papers100M(data_dir):
    ogb_dataset = PygNodePropPredDataset('ogbn-papers100M', root=data_dir)
    ogb_data = ogb_dataset[0]
    dataset = NCDataset('ogbn-papers100M')
    dataset.graph = dict()
    dataset.graph['edge_index'] = torch.as_tensor(ogb_data.edge_index)
    dataset.graph['node_feat'] = torch.as_tensor(ogb_data.x)
    dataset.graph['num_nodes'] = ogb_data.num_nodes

    # Use mapped train, valid and test index, same as OGB.
    split_idx = ogb_dataset.get_idx_split()
    train_idx, valid_idx, test_idx = split_idx['train'], split_idx['valid'], split_idx['test']

    # dataset.label = torch.as_tensor(ogb_data.y.data[all_idx], dtype=int).reshape(-1, 1)
    dataset.label = torch.as_tensor(ogb_data.y.data, dtype=int).reshape(-1, 1)  # 99% labels are nan, not available

    # print(f'f1:{dataset.label.shape}')
    # print(f'{ogb_data.num_nodes}')
    def get_idx_split():
        split_idx = {
            'train': train_idx,
            'valid': valid_idx,
            'test': test_idx,
        }
        return split_idx

    dataset.load_fixed_splits = get_idx_split

    return dataset


def load_pokec_mat(data_dir):
    """ requires pokec.mat """
    if not path.exists(f'{data_dir}/pokec/pokec.mat'):
        gdd.download_file_from_google_drive(
            file_id=dataset_drive_url['pokec'],
            dest_path=f'{data_dir}/pokec/pokec.mat',
            showsize=True
        )

    try:
        fulldata = scipy.io.loadmat(f'{data_dir}/pokec/pokec.mat')
        edge_index = fulldata['edge_index']
        node_feat = fulldata['node_feat']
        label = fulldata['label']
    except:
        edge_index = np.load(f'{data_dir}/pokec/edge_index.npy')
        node_feat = np.load(f'{data_dir}/pokec/node_feat.npy')
        label = np.load(f'{data_dir}/pokec/label.npy')

    dataset = NCDataset('pokec')
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    node_feat = torch.tensor(node_feat).float()
    num_nodes = int(node_feat.shape[0])
    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': node_feat,
                     'num_nodes': num_nodes}

    label = torch.tensor(label).flatten()
    dataset.label = torch.tensor(label, dtype=torch.long)

    def load_fixed_splits(train_prop=0.5, val_prop=0.25):
        dir = f'{data_dir}pokec/split_0.5_0.25'
        tensor_split_idx = {}
        if os.path.exists(dir):
            tensor_split_idx['train'] = torch.as_tensor(np.loadtxt(dir + '/pokec_train.txt'), dtype=torch.long)
            tensor_split_idx['valid'] = torch.as_tensor(np.loadtxt(dir + '/pokec_valid.txt'), dtype=torch.long)
            tensor_split_idx['test'] = torch.as_tensor(np.loadtxt(dir + '/pokec_test.txt'), dtype=torch.long)
        else:
            os.makedirs(dir)
            tensor_split_idx['train'], tensor_split_idx['valid'], tensor_split_idx['test'] \
                = rand_train_test_idx(dataset.label, train_prop=train_prop, valid_prop=val_prop)
            np.savetxt(dir + '/pokec_train.txt', tensor_split_idx['train'], fmt='%d')
            np.savetxt(dir + '/pokec_valid.txt', tensor_split_idx['valid'], fmt='%d')
            np.savetxt(dir + '/pokec_test.txt', tensor_split_idx['test'], fmt='%d')
        return tensor_split_idx

    dataset.load_fixed_splits = load_fixed_splits
    return dataset


def load_snap_patents_mat(data_dir, nclass=5):
    if not path.exists(f'{data_dir}snap_patents.mat'):
        p = dataset_drive_url['snap-patents']
        print(f"Snap patents url: {p}")
        gdd.download_file_from_google_drive(
            file_id=dataset_drive_url['snap-patents'], \
            dest_path=f'{data_dir}snap_patents.mat', showsize=True)

    fulldata = scipy.io.loadmat(f'{data_dir}snap_patents.mat')

    dataset = NCDataset('snap_patents')
    edge_index = torch.tensor(fulldata['edge_index'], dtype=torch.long)
    node_feat = torch.tensor(
        fulldata['node_feat'].todense(), dtype=torch.float)
    num_nodes = int(fulldata['num_nodes'])
    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': node_feat,
                     'num_nodes': num_nodes}

    years = fulldata['years'].flatten()
    label = even_quantile_labels(years, nclass, verbose=False)
    dataset.label = torch.tensor(label, dtype=torch.long)

    return dataset


def papers100M_sub(data_dir):
    data_path = os.path.join(data_dir, 'ogbn_papers100M', 'subgraph.pt')

    num_nodes = 1000000
    ogb_dataset = PygNodePropPredDataset('ogbn-papers100M', root=data_dir)
    ogb_data = ogb_dataset[0]
    edge_index = torch.as_tensor(ogb_data.edge_index)
    node_feat = torch.as_tensor(ogb_data.x)
    total_nodes = ogb_data.num_nodes
    node_labels = torch.as_tensor(ogb_data.y.data, dtype=int).reshape(-1, 1)

    split_idx = ogb_dataset.get_idx_split()
    train_idx, valid_idx, test_idx = split_idx['train'], split_idx['valid'], split_idx['test']

    # print(f'train:{train_idx.min()}, {train_idx.max()}, {train_idx.shape}')
    # print(f'valid:{valid_idx.min()}, {valid_idx.max()}, {valid_idx.shape}')
    # print(f'test:{test_idx.min()}, {test_idx.max()}, {test_idx.shape}')

    train_idx_i = train_idx[train_idx < num_nodes]
    valid_idx_i = valid_idx[valid_idx < num_nodes]
    test_idx_i = test_idx[test_idx < num_nodes]
    split_idx = torch.cat([train_idx_i, valid_idx_i, test_idx_i])
    split_len = split_idx.shape[0]
    train_num = int(split_len * 0.7)
    valid_num = int(split_len * 0.1)
    train_idx_i = split_idx[:train_num]
    valid_idx_i = split_idx[train_num:train_num + valid_num]
    test_idx_i = split_idx[train_num + valid_num:]

    idx_i = torch.arange(num_nodes)

    if os.path.exists(data_path):
        edge_index_i = torch.load(data_path)
        # f=open(data_path,'rb')
        # edge_index_i=pkl.load(f)
    else:
        edge_index_i, _ = subgraph(idx_i, edge_index, num_nodes=total_nodes, relabel_nodes=False)

    x_i = node_feat[:num_nodes]
    y_i = node_labels[:num_nodes]

    print(f'train new: {train_idx_i.shape}')
    print(f'valid new: {valid_idx_i.shape}')
    print(f'test new: {test_idx_i.shape}')

    dataset = NCDataset('ogbn-papers100M')
    dataset.graph = dict()
    dataset.graph['edge_index'] = edge_index_i
    dataset.graph['node_feat'] = x_i
    dataset.graph['num_nodes'] = num_nodes
    dataset.label = y_i

    def get_idx_split():
        split_idx = {
            'train': train_idx_i,
            'valid': valid_idx_i,
            'test': test_idx_i,
        }
        return split_idx

    dataset.load_fixed_splits = get_idx_split

    folder = os.path.join(data_dir, 'ogbn_papers100M')
    if not os.path.exists(folder):
        os.mkdir(folder)
    if not os.path.exists(data_path):
        torch.save(edge_index_i, data_path)
    # f=open(data_path,'wb')
    # pkl.dump(edge_index_i,f)

    return dataset


if __name__ == '__main__':
    # load_papers100M('../data')
    papers100M_sub('../../data')
