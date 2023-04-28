import torch
import torch.nn as nn
from GCN.settings import config

class GraphGather(nn.Module):
    def forward(self, x, feature_size_list):
        x_gataher = torch.zeros((0, x.shape[1]), dtype=torch.float).to(config["device"])
        for i, feature_size in enumerate(feature_size_list):
            start = sum(feature_size_list[:i])
            end = start + feature_size
            x_gataher = torch.cat([x_gataher, torch.mean(x[start:end, :], dim=0).view(1, -1)])
        return x_gataher
        


class TanhExp(nn.Module):
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * torch.tanh(torch.exp(x))