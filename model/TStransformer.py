from torch import nn
import torch
from utils.conf import Configuration


class PositionalEncoding(nn.Module):
    def __init__(self, conf: Configuration, dropout = 0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.max_len = conf.getEntry('len_series')
        self.embed_size = conf.getEntry('embed_size')
        self.device = conf.getEntry('device')

        position = torch.arange(self.max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embed_size, 2).float() * -(torch.log(torch.tensor(10000.0)) / self.embed_size))
        pos_encoding = torch.zeros(self.max_len, self.embed_size)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pos_encoding', pos_encoding)

    def forward(self, x):
        x = x + self.pos_encoding[:x.size(1), :]
        return self.dropout(x)
    

class TStransformer(nn.Module):
    def __init__(self, conf: Configuration):
        super(TStransformer, self).__init__()
        
        self.conf = conf
        self.dim_series = conf.getEntry("dim_series")
        self.len_series = conf.getEntry("len_series")
        self.len_reduce = conf.getEntry("len_reduce")
        self.embed_size = conf.getEntry("embed_size")
        self.num_heads = conf.getEntry("num_heads")
        self.num_layers = conf.getEntry("num_layers")
        self.batch_size = conf.getEntry("batch_size")
        
        self.embedding = nn.Linear(self.dim_series, self.embed_size)
        self.positional_encoding = PositionalEncoding(self.conf)
        self.transformer = nn.Transformer(self.embed_size, self.num_heads, self.num_layers)
        self.reduce = nn.Linear(self.len_series, self.len_reduce)
        self.de_embedding = nn.Linear(self.embed_size, self.dim_series)

    def forward(self, x):
        # x: (batch_size, dim_series, len_series)
        #print("1", x.shape)
        
        x = x.permute(0, 2, 1)  # (batch_size, len_series, dim_series)
        x = self.embedding(x)  # (batch_size, len_series, embed_size)
        #print("2", x.shape)
        
        x = self.positional_encoding(x.permute(1, 0, 2))  # (len_series, batch_size, embed_size)
        #print("3", x.shape)
        
        x = self.transformer(x, x)  # (len_series, batch_size, embed_size)
        #print("4", x.shape)
        
        x = x.permute(2, 1, 0)  # (embed_size, batch_size, len_series)
        #print("5", x.shape)
        
        x = x.reshape(-1, self.len_series)  # (embde_size*batch_size, len_series) 
        #print("6", x.shape)
        
        x = self.reduce(x)  # (embed_size*batch_size, len_reduce)
        #print("7", x.shape)
        
        x = x.reshape(self.embed_size, -1, self.len_reduce)  # (embed_size, batch_size, len_reduce)
        #print("8", x.shape)
        
        x = x.permute(1, 2, 0)  # (batch_size, len_reduce, embed_size)
        #print("9", x.shape)
        
        x = x.reshape(-1, self.embed_size)  # (batch_size*len_reduce, embed_size)
        #print("10", x.shape)
        
        x = self.de_embedding(x)  # (batch_size*len_reduce, dim_series)
        #print("11", x.shape)
        
        x = x.reshape(-1, self.len_reduce, self.dim_series)  # (batch_size, len_reduce, dim_series)
        #print("12", x.shape)
        
        x = x.permute(0, 2, 1)  # (batch_size, dim_series, len_reduce)
        #print("13", x.shape)
        
        return x
