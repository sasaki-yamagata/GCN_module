import numpy as np
import os
import pandas as pd
import pytz
import torch
from datetime import datetime
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from GCN.data.dataloader import DataFrameLoader
from GCN.data.dataset import MoleculeDataset, gcn_collate_fn
from GCN.models.graph_conv_model import GraphConvModel
from GCN.common.method import fit, predict, accuracy ,torch_seed
from GCN.settings import config

def main(df_path, target_props):
    torch_seed()
    loader = DataFrameLoader(df_path=df_path, target_props=target_props, start=config["start"], end=config["end"])

    x_train, x_test, y_train, y_test = train_test_split(loader.x_data,  
                                                        loader.y,
                                                        test_size=0.25, 
                                                        random_state=1)
    del loader
    molecule_dataset_train = MoleculeDataset(x_train, y_train)
    molecule_dataset_test = MoleculeDataset(x_test, y_test)
    del x_train, y_train, x_test, y_test

    n_input = molecule_dataset_train.x_data[0].x.shape[1]
    n_output = molecule_dataset_train.y.shape[1]

    data_loader_train = DataLoader(molecule_dataset_train, batch_size=config["batch_size"], collate_fn=gcn_collate_fn)
    data_loader_test = DataLoader(molecule_dataset_test, batch_size=config["batch_size"], collate_fn=gcn_collate_fn)
    del molecule_dataset_train, molecule_dataset_test

    gc_hidden_size_list = [
        2**config["node_num"] if i % 2 == 0 else 2**config["node_num"]*2 for i in range(config["gc_layer_num"])]
    affine_hidden_size_list = [
        2**config["node_num"]*2 if i % 2 == 0 else 2**config["node_num"] for i in range(config["affine_layer_num"])] + [config["last_layer_node_num"]]
    
    model = GraphConvModel(n_input, n_output, gc_hidden_size_list, affine_hidden_size_list).to(config["device"])


    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=config["lr"])


    # 学習
    model, history = fit(model, optimizer, criterion, config["n_epoch"], data_loader_train, data_loader_test)


    # 保存
    save_folder = f"result/{datetime.now(pytz.timezone('Asia/Tokyo')).strftime('%Y-%m-%d-%H%M%S')}"
    os.makedirs(save_folder, exist_ok=True)
    target_props_str = "_".join(target_props)
    np.save(f"{save_folder}/config.npy", config)
    torch.save(model, f'{save_folder}/model_{target_props_str}.pth')
    torch.save(model.state_dict(), f"{save_folder}/model_weight_{target_props_str}.pth")
    df_history = pd.DataFrame(history)
    df_history.to_csv(f"{save_folder}/history_{target_props_str}.csv")

    # 予測
    result_train, result_test = predict(model, data_loader_train, data_loader_test)

    # 保存
    train_columns = [f"train_obs_{target_prop}" for target_prop in target_props] \
                        + [f"train_pre_{target_prop}" for target_prop in target_props ]
    test_columns = [f"test_obs_{target_prop}" for target_prop in target_props] \
                        + [f"test_pre_{target_prop}" for target_prop in target_props ] 
    df_result_train = pd.DataFrame(result_train.detach().to("cpu"), columns=train_columns)
    df_result_test = pd.DataFrame(result_test.detach().to("cpu"), columns=test_columns)
    df_result_train.to_csv(f"{save_folder}/result_train_{target_props_str}.csv")
    df_result_test.to_csv(f"{save_folder}/result_test_{target_props_str}.csv")

    # 精度
    df_accuracy = accuracy(target_props, result_train, result_test)
    df_accuracy.to_csv(f"{save_folder}/accuracy_{target_props_str}.csv")


if __name__ == "__main__":
    abs_path = os.path.dirname(os.path.abspath(__file__))
    main(df_path=f"{abs_path}/datasets/homolumo_matrix.pkl", 
         target_props=["HOMO", "LUMO"])


