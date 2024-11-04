import json

class Configuration:
    def __init__(self, confPath: str = ""):
        self.confPath = confPath
        self.defaultConf = {
            "data_path": "../Data/Mine/gist/gist.data",
            "log_path": "./example/example.log",
            "model_path": "./example/example.pth",
            "train_path": "./example/data/train.data",
            "val_path": "./example/data/val.data",
            "train_indices_path": "./example/data/train_indices.data",
            "val_indices_path": "./example/data/val_indices.data",

            "epoch_max": 100,
            "device": "cuda:6",
            "len_series": 960,
            "len_reduce": 60,
            "dim_series": 1,
            "batch_size": 16,
            
            "data_size": 1000000,
            "train_size": 20000,
            "val_size": 1000,

            "embed_size": 16,
            "num_tokens": 10000,
            "num_heads": 4,
            "num_layers": 6,

            "orth_regularizer": "srip",
            "srip_mode": "linear",
            "srip_max": 5e-4,
            "srip_min": 0
        }
        self.loadConf()
    
    
    def loadConf(self):
        with open(self.confPath, 'r') as fin:
            self.confLoaded = json.load(fin)
            
            
    def getEntry(self, key: str):
        if key in self.confLoaded:
            return self.confLoaded[key]
        elif key in self.defaultConf:
            return self.defaultConf[key]
        else:
            raise Exception(f"Key {key} not found in configuration.")