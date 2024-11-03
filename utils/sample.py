from utils.conf import Configuration
import numpy as np
from torch.utils.data import Dataset
import torch


class TSData(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    

def getSample(conf: Configuration):
    data_path = conf.getEntry('data_path')
    train_path = conf.getEntry('train_path')
    train_indices_path = conf.getEntry('train_indices_path')
    val_indices_path = conf.getEntry('val_indices_path')
    val_path = conf.getEntry('val_path')
    
    len_series = conf.getEntry('len_series')
    
    data_size = conf.getEntry('data_size')
    train_size = conf.getEntry('train_size')
    val_size = conf.getEntry('val_size')
    
    train_samples_indices = np.random.randint(0, data_size, size = train_size, dtype=np.int64)
    val_samples_indices = np.random.randint(0, data_size, size = val_size, dtype=np.int64)
    
    train_samples_indices.tofile(train_indices_path)
    val_samples_indices.tofile(val_indices_path)
    
    loaded = []
    for index in train_samples_indices:
        sequence = np.fromfile(data_path, dtype = np.float32, count = len_series, offset = 4 * len_series * index)
        if not np.isnan(np.sum(sequence)):
            loaded.append(sequence)

    # 二维 numpy 数组
    train_samples = np.array(loaded, dtype=np.float32)
    train_samples.tofile(train_path)
    # 三维 torch 张量
    train_samples = torch.from_numpy(train_samples).view([-1, 1, len_series])
    
    loaded = []
    for index in val_samples_indices:
        sequence = np.fromfile(data_path, dtype = np.float32, count = len_series, offset = 4 * len_series * index)
        if not np.isnan(np.sum(sequence)):
            loaded.append(sequence)
    
    val_samples = np.asarray(loaded, dtype=np.float32)
    val_samples.tofile(val_path)
    val_samples = torch.from_numpy(val_samples).view([-1, 1, len_series])
    
    return train_samples, val_samples