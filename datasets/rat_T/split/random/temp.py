# !/usr/bin/env python3
# _*_ coding:utf-8 _*_
"""
@File     : temp.py
@Project  : gnn_base_model
@Time     : 2022/12/9 14:19
@Author   : Pu Chengtao
@Contact_2: 2319189860@qq.com
@Software : PyCharm
@Last Modify Time      @Version     @Desciption
--------------------       --------        -----------
2022/12/9 14:19        1.0             None
"""
import torch

def train_val_test_idx(num_data,train_size=0.8,val_size=0.1,seed=None):
    if seed:
        np.random.seed(seed)
    idx = np.arange(num_data)
    np.random.shuffle(idx)
    train_idx = idx[:np.int(np.ceil(train_size*num_data))]
    valid_idx = idx[np.int(np.ceil(train_size*num_data)):np.int(np.ceil((train_size+val_size)*num_data))]
    test_idx = idx[np.int(np.ceil((train_size+val_size)*num_data)):]

    return {'train':torch.from_numpy(train_idx).to(torch.long),'valid':torch.from_numpy(valid_idx).to(torch.long),'test':torch.from_numpy(test_idx).to}