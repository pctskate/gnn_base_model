# !/usr/bin/env python3
# _*_ coding:utf-8 _*_
"""
@File     : main.py
@Project  : gnn_base_model
@Time     : 2022/12/8 17:13
@Author   : Pu Chengtao
@Contact_2: 2319189860@qq.com
@Software : PyCharm
@Last Modify Time      @Version     @Desciption
--------------------       --------        -----------
2022/12/8 17:13        1.0             None
"""
import torch
from torch_geometric.loader import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from gnn import GNN

from tqdm import tqdm
import argparse
import time
import numpy as np

from dataset2 import MolDataset
from evaluate import Evaluator

cls_criterion = torch.nn.BCEWithLogitsLoss()
reg_criterion = torch.nn.MSELoss()


def train(model, device, loader, optimizer, task_type):
    model.train()
    batch_loss = []
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            pred = model(batch)
            optimizer.zero_grad()
            ## ignore nan targets (unlabeled) when computing training loss.
            is_labeled = batch.y == batch.y
            if "classification" in task_type:
                loss = cls_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            else:
                loss = reg_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())

    return sum(batch_loss)/len(batch_loss)


def valid(model,device,loader,task_type):
    model.eval()
    batch_loss = []
    for step,batch in enumerate(tqdm(loader,desc='Iteration')):
        batch = batch.to(device)
        with torch.no_grad():
            pred = model(batch)

            is_labeled = batch.y == batch.y
            if "classification" in task_type:
                loss = cls_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            else:
                loss = reg_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            batch_loss.append(loss.item())

    return sum(batch_loss) / len(batch_loss)


def eval(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(batch)

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='GNN baselines on ogbgmol* data with Pytorch Geometrics')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--gnn', type=str, default='gin-virtual',
                        help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
    parser.add_argument('--drop_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    # parser.add_argument('--dataset', type=str, default="ogbg-molhiv",
    #                     help='dataset name (default: ogbg-molhiv)')

    # parser.add_argument('--feature', type=str, default="full",
    #                     help='full feature or simple feature')
    parser.add_argument('--load_pretrained',type=bool,default=False,
                        help='load from pretrained model')
    parser.add_argument('--model_path',type=str,default="",
                        )
    parser.add_argument('--filename', type=str, default="",
                        help='filename to output result (default: )')
    args = parser.parse_args()

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    ### automatic dataloading and splitting
    root = './datasets'
    task_dict = {'task_name': 'rat_T',
                 'task_type': 'reg',
                 'num_tasks': 1,
                 }
    dataset = MolDataset(root,task_dict)

    split_idx = dataset.get_idx_split()

    # To do: when save 'split.pt',specify the dtype np.int64
    train_idx = split_idx["train"].astype(np.int64)
    valid_idx = split_idx['valid'].astype(np.int64)
    test_idx = split_idx['test'].astype(np.int64)
    train_loader = DataLoader(dataset[train_idx], batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    valid_loader = DataLoader(dataset[valid_idx], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    test_loader = DataLoader(dataset[test_idx], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)

    if args.gnn == 'gin':
        model = GNN(gnn_type = 'gin', num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False).to(device)
    elif args.gnn == 'gin-virtual':
        model = GNN(gnn_type = 'gin', num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = True).to(device)
    elif args.gnn == 'gcn':
        model = GNN(gnn_type = 'gcn', num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False).to(device)
    elif args.gnn == 'gcn-virtual':
        model = GNN(gnn_type = 'gcn', num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = True).to(device)
    else:
        raise ValueError('Invalid GNN type')

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    if args.load_pretrained == True and not args.model_path == "":
        model.load_state_dict(torch.load(args.model_path))
    valid_loss = 1e3
    writer = SummaryWriter()
    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}".format(epoch))
        print('Training...')
        epoch_train_loss = train(model, device, train_loader, optimizer, dataset.task_type)
        writer.add_scalar("Loss/train",epoch_train_loss,epoch)

        epoch_valid_loss = valid(model,device,valid_loader,dataset.task_type)
        writer.add_scalar("Loss/valid",epoch_valid_loss,epoch)

        if epoch_valid_loss <= valid_loss:
            valid_loss = epoch_valid_loss
            print(f'Current best valid loss : {valid_loss}')
            torch.save(model.state_dict(),'./experiments/exp1/best-model.pth')

        # print('Evaluating...')
        # train_perf = eval(model, device, train_loader, evaluator)
        # valid_perf = eval(model, device, valid_loader, evaluator)
        # test_perf = eval(model, device, test_loader, evaluator)
        #
        # print({'Train': train_perf, 'Validation': valid_perf, 'Test': test_perf})
        #
        # train_curve.append(train_perf[dataset.eval_metric])
        # valid_curve.append(valid_perf[dataset.eval_metric])
        # test_curve.append(test_perf[dataset.eval_metric])

    print(f"{20*'*'}Finished training!{20*'*'}")
    print(f"{20*'*'}Loading the best model.......{20*'*'}")
    model.load_state_dict(torch.load('./experiments/exp1/best-model.pth'))
    print(f"{20 * '*'}Loading best model successfully{20 * '*'}")
    print('Evaluating......')
    ### automatic evaluator. takes dataset name as input
    eval_dict = {
        'task_name':'rat_T',
        'num_tasks':1,
        'eval_metric':'reg'
    }
    evaluator = Evaluator(eval_dict)
    train_reg_dic = eval(model,device,train_loader,evaluator)
    valid_reg_dic = eval(model,device,valid_loader,evaluator)
    test_reg_dic = eval(model,device,test_loader,evaluator)
    print(f"{20 * '*'}The result in test set:{20 * '*'}")
    for key,value in test_reg_dic.items():
        # 输出测试集的评估结果
        print(f"{20 * '*'}{key} : {value}{20 * '*'}")

    # 保存训练、验证和测试的结果
    if not args.filename == '':
        with open(args.filename,'w') as f:
            print(f'All args : {vars(args)}\n')
            f.write(f'Train set result : {train_reg_dic}\n')
            f.write(f'Valid set result : {valid_reg_dic}\n')
            f.write(f'Test set result : {test_reg_dic}\n')


if __name__ == "__main__":
    main()