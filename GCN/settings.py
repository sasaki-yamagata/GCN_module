import torch

def get_config(debug=False):
    """
    GCNのハイパーパラメータの設定を取得

    Parameters
    ----------
    debug : boolean
        Trueにするとデバックモードにできる
    
    Returns
    ----------
    config : dict
        GCNのハイパーパラメータ

    Notes
    ----------
    configについて
        start : integer
            抽出するデータの最初を指定
        end : integer
            抽出するデータの最後を指定
        device : string
            cpuとgpuのどちらで計算するかを指定
        n_epoch : integer
            エポック数を指定
        lr : float
            学習率を指定
        batch_size : integer
            バッチサイズを指定
        n_splits : integer
            交差検証する際の分割数を指定
    """
    
    if debug:
        config = {
            "start": None,
            "end": 1000,
            "device": 'cuda' if torch.cuda.is_available() else 'cpu',
            "n_epoch": 1,
            "lr": 0.001,
            "batch_size" : 10,
            "n_splits": 2, # 交差検証のときのみ使用  
            "gc_layer_num" : 3,
            "affine_layer_num" : 2,
            "node_num" : 4,
            "last_layer_node_num" : 6,
        }
        
    else:
        config = {
            "start": None,
            "end": None,
            "device": 'cuda' if torch.cuda.is_available() else 'cpu',
            "n_epoch": 50,
            "lr": 0.001,
            "batch_size" : 50,
            "n_splits": 3 # 交差検証のときのみ使用  
        }
        

    print(f"-------------  Device in this enviroment is {config['device']} -------------")
    return config

config = get_config(debug=True)




