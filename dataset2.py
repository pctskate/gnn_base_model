# !/usr/bin/env python3
# _*_ coding:utf-8 _*_
"""
@File     : dataset2.py
@Project  : gnn_base_model
@Time     : 2022/12/9 9:58
@Author   : Pu Chengtao
@Contact_2: 2319189860@qq.com
@Software : PyCharm
@Last Modify Time      @Version     @Desciption
--------------------       --------        -----------
2022/12/9 9:58        1.0             None
"""
import torch
import numpy as np
import os
import os.path as osp
from torch_geometric.data import InMemoryDataset
from utils.read_graph_pyg import read_graph_pyg


class MolDataset(InMemoryDataset):
    def __init__(self,root,task_dict,transform=None, pre_transform=None):
        self.task_name = task_dict['task_name']
        self.task_type = task_dict['task_type']
        self.num_tasks = task_dict['num_tasks']
        self.root = osp.join(root,task_dict['task_name'])

        super(MolDataset, self).__init__(self.root, transform, pre_transform)

        self.data,self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['data.npz']

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    @property
    def num_classes(self):
        return self.num_tasks

    def get_idx_split(self,split_type='random'):
        path = osp.join(self.root,'split',split_type)

        return torch.load(os.path.join(path,'split_dict.pt'))

    def process(self):
        add_inverse_edge = True
        additional_node_files = []
        additional_edge_files = []

        data_list = read_graph_pyg(self.raw_dir, binary=True)
        graph_label = np.load(osp.join(self.raw_dir, 'graph-label.npz'))['graph_label']

        has_nan = np.isnan(graph_label).any()
        for i, g in enumerate(data_list):
            if 'classification' in self.task_type:
                if has_nan:
                    g.y = torch.from_numpy(graph_label[i]).view(1, -1).to(torch.float32)
                else:
                    g.y = torch.from_numpy(graph_label[i]).view(1, -1).to(torch.long)
            else:
                g.y = torch.from_numpy(graph_label[i]).view(1, -1).to(torch.float32)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])


if __name__ == '__main__':
    root = './datasets'
    task_dict = {'task_name':'rat_T',
                 'task_type':'reg',
                 'num_tasks':1,
                 }
    dataset = MolDataset(root,task_dict)

