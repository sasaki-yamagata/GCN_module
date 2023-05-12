import torch
from datetime import datetime

from torch import nn
from torch_geometric.nn import GCNConv, Sequential
from GCN.settings import config

from ..layers.graph_conv_layer import GraphGather, TanhExp
# logging.basicConfig(level=logging.INFO, filename=f"logs/{datetime.now().strftime('%Y-%m-%d-%H%M%S')}outputs.log", format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
class GraphConvModel(nn.Module):
    def __init__(self, n_input, n_output, gc_hidden_size_list, affine_hidden_size_list):
        super().__init__()
        activation = TanhExp.apply
        conv_layers = []
        for i in range(len(gc_hidden_size_list)):
            if i == 0:
                conv_layers.append((GCNConv(n_input, gc_hidden_size_list[i]), "x, edge_index -> x"))
                conv_layers.append(activation)
            else:
                conv_layers.append((GCNConv(gc_hidden_size_list[i-1], gc_hidden_size_list[i]), "x, edge_index -> x"))
                conv_layers.append(activation)
                # setattr(self, f"conv{num}", GCNConv(gc_hidden_size_list[i-1], gc_hidden_size_list[i]))
        liner_layers = []
        for i in range(len(affine_hidden_size_list)):
            if i == 0:
                liner_layers.append((nn.Linear(gc_hidden_size_list[-1], affine_hidden_size_list[i]), "x -> x"))
                liner_layers.append(activation)
                # setattr(self, f"l{num}", nn.Linear(gc_hidden_size_list[-1], affine_hidden_size_list[i]))

            elif i == len(affine_hidden_size_list) - 1:
                liner_layers.append((nn.Linear(affine_hidden_size_list[i-1], affine_hidden_size_list[i]), "x -> x"))
                liner_layers.append(activation)
                liner_layers.append((nn.Linear(affine_hidden_size_list[i], n_output), "x -> x"))

                # setattr(self, f"l{num}", nn.Linear(affine_hidden_size_list[i-1], affine_hidden_size_list[i]))
                # setattr(self, f"l{num}", nn.Linear(affine_hidden_size_list[i], n_output))
            else:
                liner_layers.append((nn.Linear(affine_hidden_size_list[i-1], affine_hidden_size_list[i]), "x -> x"))
                liner_layers.append(activation)
                # setattr(self, f"l{num}", nn.Linear(affine_hidden_size_list[i-1], affine_hidden_size_list[i]))
        
        
        self.conv = Sequential("x, edge_index", conv_layers)
        self.gather = GraphGather()
        self.linear = Sequential("x", liner_layers)
        # self.conv = nn.Sequential(
        #     self.conv1,
        #     self.tanhexp, 
        #     self.conv2,
        #     self.tanhexp,
        #     self.conv3,
        #     self.tanhexp
        # )

        # self.linear = nn.Sequential(
        #     self.l1,
        #     self.tanhexp,
        #     self.l2,
        #     self.tanhexp,
        #     self.l3,
        # )



    def forward(self, x, edge_index, feature_size_list):
        x = self.conv(x, edge_index)
        x = self.gather(x, feature_size_list)
        x = self.linear(x)
        # # logging.info(f"conv1{x}")
        # x = self.tanhexp(x)
        # x = self.conv2(x, edge_index)
        # # logging.info(f"conv2{x}")
        # x = self.tanhexp(x)
        # x = self.conv3(x, edge_index)
        # # logging.info(f"conv3{x}")
        # x = self.tanhexp(x)
        # x = self.gather(x, feature_size_list)
        # # logging.info(f"gather{x}")
        # x = self.l1(x)
        # # logging.info(f"l1{x}")
        # x = self.tanhexp(x)
        # x = self.l2(x)
        # # logging.info(f"l2{x}")
        # x = self.tanhexp(x)
        # x = self.l3(x)
        # # logging.info(f"l3{x}")
        return x
    