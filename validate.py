import numpy as np
import optuna
import os
import pickle
import pytz
from datetime import datetime
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split, KFold

from GCN.data.dataloader import DataFrameLoader
from GCN.data.dataset import MoleculeDataset, gcn_collate_fn
from GCN.models.graph_conv_model import GraphConvModel
from GCN.common.method import fit, predict, accuracy ,torch_seed
from GCN.settings import config


def main(df_path, target_props, timeout, study_path=None):
    torch_seed()
    loader = DataFrameLoader(df_path=df_path, target_props=target_props, start=config["start"], end=config["end"])

    n_input = loader.x_data[0].x.shape[1]
    n_output = len(target_props)

    x_train_val, _, y_train_val, _, molcode_train, _ = train_test_split(loader.x_data,  
                                                    loader.y,
                                                    loader.molcode,
                                                    test_size=0.25, 
                                                    random_state=1)
    
    if study_path is None:
        study = optuna.create_study(direction="minimize")
    else:
        with open(study_path, "rb") as f:
            study = pickle.load(f)

    study.optimize(lambda trial: objective(trial, x_train_val, y_train_val, n_input, n_output, target_props), catch=(ValueError,), timeout=timeout)
    print(study.best_params)

    # 保存
    save_folder = f"valid_result/{datetime.now(pytz.timezone('Asia/Tokyo')).strftime('%Y-%m-%d-%H%M%S')}"
    os.makedirs(save_folder, exist_ok=True)
    target_props_str = "_".join(target_props)
    study.trials_dataframe().to_csv(f"{save_folder}/validate_{target_props_str}.csv")
    with open(f"{save_folder}/study_{target_props_str}.pkl", "wb") as f:
        pickle.dump(study, f)
    with open(f"{save_folder}/molcode_train_{target_props_str}.pkl", "wb") as f:
        pickle.dump(molcode_train, f)
    

def objective(trial, x, y, n_input, n_output, target_props):

    bayes_params = {
        "gc_layer_num" : trial.suggest_int("gc_layer_num", 1, 9),
        "affine_layer_num" : trial.suggest_int("affine_layer_num", 1, 9),
        "node_num" : trial.suggest_int("node_num", 5, 7),
        "last_layer_node_num" : trial.suggest_int("last_layer_node_num", 4, 10),
        "lr" : trial.suggest_float("lr", 1e-4, 1e-3)
    }

    config.update(bayes_params)

    gc_hidden_size_list = [
        2**config["node_num"] if i % 2 == 0 else 2**config["node_num"]*2 for i in range(config["gc_layer_num"])]
    affine_hidden_size_list = [
        2**config["node_num"]*2 if i % 2 == 0 else 2**config["node_num"] for i in range(config["affine_layer_num"])] + [config["last_layer_node_num"]]
    
    metrix = "rmse" # ベイズ最適化の評価指標を指定
    kf = KFold(n_splits=config["n_splits"])
    for i, (train, test) in enumerate(kf.split(x)):
        x_train = [x[tr] for tr in train]
        x_test = [x[ts] for ts in test]
        y_train = y[train]
        y_test = y[test]
        molecule_dataset_train = MoleculeDataset(x_train, y_train)
        molecule_dataset_test = MoleculeDataset(x_test, y_test)
        data_loader_train = DataLoader(molecule_dataset_train, batch_size=config["batch_size"], collate_fn=gcn_collate_fn)
        data_loader_test = DataLoader(molecule_dataset_test, batch_size=config["batch_size"], collate_fn=gcn_collate_fn)
        model = GraphConvModel(n_input, n_output, gc_hidden_size_list, affine_hidden_size_list).to(config["device"])

        criterion = nn.MSELoss()
        
        optimizer = optim.Adam(model.parameters(), lr=config["lr"])
        model, _ = fit(model, optimizer, criterion, config["n_epoch"], data_loader_train, data_loader_test, is_detect_anomaly=True)

        result_train, result_test = predict(model, data_loader_train, data_loader_test)
        df_accuracy_temp = accuracy(target_props, result_train, result_test)
        if i == 0:
            df_accuracy_sum = df_accuracy_temp
        else:
            df_accuracy_sum = df_accuracy_sum.add(df_accuracy_temp)

    df_accuracy = df_accuracy_sum / config["n_splits"]
    print(df_accuracy)
    # 保存
    # save_folder = f"valid_result/{datetime.now(pytz.timezone('Asia/Tokyo')).strftime('%Y-%m-%d-%H%M%S')}"
    # os.makedirs(save_folder, exist_ok=True)
    # target_props_str = "_".join(target_props)
    # np.save(f"{save_folder}/config.npy", config)
    # df_accuracy.to_csv(f"{save_folder}/accuracy_avg_{target_props_str}.csv")
    df_test_accuracy = df_accuracy.loc[[i for i in df_accuracy.index if "test" in i], metrix]
    valid_value = df_test_accuracy.mean()
    return valid_value

if __name__ == "__main__":
    abs_path = os.path.dirname(os.path.abspath(__file__))
    main(df_path=f"{abs_path}/datasets/homolumo_matrix.pkl", 
         target_props=["HOMO", "LUMO"], 
         timeout= 28800,
         study_path=f"{abs_path}/valid_result/2023-04-27-190431/study_HOMO_LUMO.pkl"
         )