# !/usr/bin/env python3
# _*_ coding:utf-8 _*_
"""
@File     : save_dataset.py
@Project  : gnn_base_model
@Time     : 2022/12/8 20:35
@Author   : Pu Chengtao
@Contact_2: 2319189860@qq.com
@Software : PyCharm
@Last Modify Time      @Version     @Desciption
--------------------       --------        -----------
2022/12/8 20:35        1.0             None
"""
import numpy as np
import os
import os.path as osp
from tqdm import tqdm
import torch
from utils.read_graph_raw import read_binary_graph_raw


def all_numpy(obj):
    # Ensure everything is in numpy or int or float (no torch tensor)

    if isinstance(obj, dict):
        for key in obj.keys():
            all_numpy(obj[key])
    elif isinstance(obj, list):
        for i in range(len(obj)):
            all_numpy(obj[i])
    else:
        if not isinstance(obj, (np.ndarray, int, float)):
            return False

    return True


class DatasetSaver(object):
    def __init__(self,root,dataset_name):
        self.dataset_name = dataset_name
        self.root = root
        self.dataset_dir = osp.join(root,self.dataset_name)

        # make necessary dirs
        self.raw_dir = osp.join(self.dataset_dir, 'raw')
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(osp.join(self.dataset_dir, 'processed'), exist_ok=True)

        # check list
        self._save_graph_list_done = False
        self._save_split_done = False
        self._save_target_labels_done = False
        self._save_task_info_done = False


    def save_graph_list(self,graph_list):
        self.num_data = len(graph_list)
        if not all_numpy(graph_list):
            raise RuntimeError('graph_list must only contain list/dict of numpy arrays, int, or float')

        dict_keys = graph_list[0].keys()
        # check necessary keys
        if not 'edge_index' in dict_keys:
            raise RuntimeError('edge_index needs to be provided in graph objects')
        if not 'num_nodes' in dict_keys:
            raise RuntimeError('num_nodes needs to be provided in graph objects')

        print(dict_keys)

        data_dict = {}
        # Store the following keys
        # - edge_index (necessary)
        # - num_nodes_list (necessary)
        # - num_edges_list (necessary)
        # - node_** (optional, node_feat is the default node features)
        # - edge_** (optional, edge_feat is the default edge features)

        # saving num_nodes_list
        num_nodes_list = np.array([graph['num_nodes'] for graph in graph_list]).astype(np.int64)
        data_dict['num_nodes_list'] = num_nodes_list

        # saving edge_index and num_edges_list
        print('Saving edge_index')
        edge_index = np.concatenate([graph['edge_index'] for graph in graph_list], axis=1).astype(np.int64)
        num_edges_list = np.array([graph['edge_index'].shape[1] for graph in graph_list]).astype(np.int64)

        if edge_index.shape[0] != 2:
            raise RuntimeError('edge_index must have shape (2, num_edges)')

        data_dict['edge_index'] = edge_index
        data_dict['num_edges_list'] = num_edges_list

        for key in dict_keys:
            if key == 'edge_index' or key == 'num_nodes':
                continue
            if graph_list[0][key] is None:
                continue

            if 'node_' == key[:5]:
                # make sure saved in np.int64 or np.float32
                dtype = np.int64 if 'int' in str(graph_list[0][key].dtype) else np.float32
                # check num_nodes
                for i in range(len(graph_list)):
                    if len(graph_list[i][key]) != num_nodes_list[i]:
                        raise RuntimeError(f'num_nodes mistmatches with {key}')

                cat_feat = np.concatenate([graph[key] for graph in graph_list], axis=0).astype(dtype)
                data_dict[key] = cat_feat

            elif 'edge_' == key[:5]:
                # make sure saved in np.int64 or np.float32
                dtype = np.int64 if 'int' in str(graph_list[0][key].dtype) else np.float32
                # check num_edges
                for i in range(len(graph_list)):
                    if len(graph_list[i][key]) != num_edges_list[i]:
                        raise RuntimeError(f'num_edges mistmatches with {key}')

                cat_feat = np.concatenate([graph[key] for graph in graph_list], axis=0).astype(dtype)
                data_dict[key] = cat_feat

            else:
                raise RuntimeError(
                    f'Keys in graph object should start from either \'node_\' or \'edge_\', but \'{key}\' given.')

        print('Saving all the files!')
        np.savez_compressed(osp.join(self.raw_dir, 'data.npz'), **data_dict)
        print('Validating...')
        # testing
        print('Reading saved files')
        graph_list_read = read_binary_graph_raw(self.raw_dir, False)

        print('Checking read graphs and given graphs are the same')
        for i in tqdm(range(len(graph_list))):
            # assert(graph_list[i].keys() == graph_list_read[i].keys())
            for key in graph_list[i].keys():
                if graph_list[i][key] is not None:
                    if isinstance(graph_list[i][key], np.ndarray):
                        assert (np.allclose(graph_list[i][key], graph_list_read[i][key], rtol=1e-4, atol=1e-4,
                                            equal_nan=True))
                    else:
                        assert (graph_list[i][key] == graph_list_read[i][key])

        del graph_list_read

        self.has_node_attr = ('node_feat' in graph_list[0]) and (graph_list[0]['node_feat'] is not None)
        self.has_edge_attr = ('edge_feat' in graph_list[0]) and (graph_list[0]['edge_feat'] is not None)
        self._save_graph_list_done = True

    def save_split(self, split_dict, split_name):
        '''
            Save dataset split
                split_dict: must contain three keys: 'train', 'valid', 'test', where the values are the split indices stored in numpy.
                split_name (str): the name of the split
        '''

        self.split_dir = osp.join(self.dataset_dir, 'split', split_name)
        os.makedirs(self.split_dir, exist_ok=True)

        # verify input
        if not 'train' in split_dict:
            raise ValueError('\'train\' needs to be given in save_split')
        if not 'valid' in split_dict:
            raise ValueError('\'valid\' needs to be given in save_split')
        if not 'test' in split_dict:
            raise ValueError('\'test\' needs to be given in save_split')

        if not all_numpy(split_dict):
            raise RuntimeError('split_dict must only contain list/dict of numpy arrays, int, or float')

        ## directly save split_dict
        ## compatible with ogb>=v1.2.3
        torch.save(split_dict, osp.join(self.split_dir, 'split_dict.pt'))

        self.split_name = split_name
        self._save_split_done = True


    def save_target_labels(self, target_labels):
        '''
            target_label (numpy.narray): storing target labels. Shape must be (num_data, num_tasks)
        '''
        if not self._save_graph_list_done:
            raise RuntimeError('save_graph_list must be done beforehand.')

        # check type and shape
        if not isinstance(target_labels, np.ndarray):
            raise ValueError(f'target label must be of type np.ndarray')
        if len(target_labels) != self.num_data:
            raise RuntimeError(
                f'The length of target_labels ({len(target_labels)}) must be the same as the number of data points ({self.num_data}).')

        # save label to graph-label.npz
        np.savez_compressed(osp.join(self.raw_dir, 'graph-label.npz'), graph_label=target_labels)
        self.num_tasks = target_labels.shape[1]

        self._save_target_labels_done = True


    def save_task_info(self, task_type, eval_metric, num_classes=None):
        '''
            task_type (str): For ogbg and ogbn, either classification or regression.
            eval_metric (str): the metric
            if task_type is 'classification', num_classes must be given.
        '''
        self.task_type = task_type

        print(self.task_type)
        print(num_classes)

        if 'classification' in self.task_type:
            if not (isinstance(num_classes, int) and num_classes > 1):
                raise ValueError(f'num_classes must be an integer larger than 1, {num_classes} given.')
            self.num_classes = num_classes
        else:
            self.num_classes = -1  # in the case of regression, just set to -1

        self.eval_metric = eval_metric

        self._save_task_info_done = True


    def check(self):
        # check everything is done before getting meta_dict
        if not self._save_graph_list_done:
            raise RuntimeError('save_graph_list not completed.')
        if not self._save_split_done:
            raise RuntimeError('save_split not completed.')
        if not self._save_target_labels_done:
            raise RuntimeError('save_target_labels not completed.')


def train_val_test_idx(num_data,train_size=0.8,val_size=0.1,seed=None):
    if seed:
        np.random.seed(seed)
    idx = np.arange(num_data)
    np.random.shuffle(idx)
    train_idx = idx[:np.int(np.ceil(train_size*num_data))]
    valid_idx = idx[np.int(np.ceil(train_size*num_data)):np.int(np.ceil((train_size+val_size)*num_data))]
    test_idx = idx[np.int(np.ceil((train_size+val_size)*num_data)):]

    return {'train':train_idx,'valid':valid_idx,'test':test_idx}


def main():
    prj_name = 'gnn_base_model'
    cur_path = os.path.abspath(os.path.dirname(__file__))
    root_path = cur_path[:cur_path.find(f"{prj_name}\\") + len(f"{prj_name}\\")]
    root = osp.join(root_path,'datasets')
    dataset_name = 'rat_T'

    # 读取smiles
    import pandas as pd
    file_name = 'T_iv.csv'
    df = pd.read_csv(osp.join(root,dataset_name,file_name))
    smiles_list = df.iloc[:,0].values
    labels = np.array(df.iloc[:,1:])

    from utils.mol import smiles2graph
    graph_list = list(map(smiles2graph,smiles_list))

    # train_test_split
    split_dict = dict()
    split_name = 'random'
    if split_name == 'scaffold':
        pass
    else:
        split_dict = train_val_test_idx(len(graph_list))

    saver = DatasetSaver(root,dataset_name)
    saver.save_graph_list(graph_list)
    saver.save_split(split_dict,split_name='random')
    saver.save_target_labels(labels)


if __name__ == '__main__':
    main()