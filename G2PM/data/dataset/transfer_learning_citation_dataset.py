import os.path as osp
import torch
import numpy as np
from torch_geometric.data import (InMemoryDataset, Data, download_url,
                                  extract_zip)
from torch_geometric.io import read_txt_array
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.data import Data, DataLoader
import numpy as np
import torch


class CitationNetworkDataset(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None, pre_filter=None):
        self.name = name
        super(CitationNetworkDataset, self).__init__(root, transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return ["docs.txt", "edgelist.txt", "labels.txt"]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):

        edge_path = osp.join(self.raw_dir, '{}_edgelist.txt'.format(self.name))
        edge_index = read_txt_array(edge_path, sep=',', dtype=torch.long).t()

        docs_path = osp.join(self.raw_dir, '{}_docs.txt'.format(self.name))
        f = open(docs_path, 'rb')
        content_list = []
        for line in f.readlines():
            line = str(line, encoding="utf-8")
            content_list.append(line.split(","))
        x = np.array(content_list, dtype=float)
        x = torch.from_numpy(x).to(torch.float)

        label_path = osp.join(self.raw_dir, '{}_labels.txt'.format(self.name))
        f = open(label_path, 'rb')
        content_list = []
        for line in f.readlines():
            line = str(line, encoding="utf-8")
            line = line.replace("\r", "").replace("\n", "")
            content_list.append(line)
        y = np.array(content_list, dtype=int)
        y = torch.from_numpy(y).to(torch.int64)

        data_list = []
        data = Data(edge_index=edge_index, x=x, y=y)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        data_list.append(data)

        data, slices = self.collate([data])

        torch.save((data, slices), self.processed_paths[0])
