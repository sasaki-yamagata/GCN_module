import torch
from rdkit import Chem
from torch.utils import data
from torch_geometric.data import Data


class MoleculeDataset(data.Dataset):
    def __init__(self, x_data, y):
        self.x_data = x_data
        self.y = y


    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self, index):
        return self.x_data[index], self.y[index, :]
    

def gcn_collate_fn(batch):

    x_data, y = zip(*batch)

    feature_batch = torch.zeros((0, x_data[0].x.shape[1]))
    edge_index_batch = torch.zeros((2, 0), dtype=torch.long)
    feature_size_list = []
    for i, x_d in enumerate(x_data):
        feature_batch = torch.cat([feature_batch, x_d.x], dim=0)
        feature_size_list.append(x_d.feature_size)
        edge_index_batch = torch.cat([edge_index_batch, x_d.edge_index+sum(feature_size_list[:i])], dim=1)

    # 目的変数をバッチ化
    y_num = y[0].shape[0]
    y_batch = torch.zeros((0, y_num), dtype=torch.float)
    for target in y:
        target = target.view(1, y_num)
        y_batch = torch.cat([y_batch, target], dim=0)

    return Data(x=feature_batch, edge_index=edge_index_batch, feature_size_list=feature_size_list), y_batch

    

    # feature_list, edge_index_list, targets = zip(*batch)

    # # 特徴量をバッチ化
    # feature_batch = torch.zeros((0, feature_list[0].shape[1]))
    # feature_sizes = []
    # for feature in feature_list:
    #     feature_batch = torch.cat([feature_batch, feature], dim=0)
    #     feature_sizes.append(feature.shape[0])

    # # edge_indexをバッチ化
    # edge_index_batch = torch.zeros((2, 0), dtype=torch.long)
    # # edge_index_sizes = []
    # batch_index = torch.zeros((0,))
    # for i, edge_index in enumerate(edge_index_list):
    #     edge_index_batch = torch.cat([edge_index_batch, edge_index+sum(feature_sizes[:i])], dim=1)
    #     edge_index_size = edge_index.shape[1]
    #     # edge_index_sizes.append(edge_index_size)
    #     batch_index = torch.cat([batch_index, torch.full((edge_index_size,), i)])

    # # 目的変数をバッチ化
    # target_num = targets[0].shape[0]
    # targets_batch = torch.zeros((0, target_num))
    # for target in targets:
    #     target = torch.tensor(target).view(1, target_num)

    #     targets_batch = torch.cat([targets_batch, target], dim=0)

    # return feature_batch, edge_index_batch, batch_index, targets_batch