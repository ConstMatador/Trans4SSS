from torch import nn
import torch
from utils.conf import Configuration


class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * -(torch.log(torch.tensor(10000.0)) / embed_size))
        pos_encoding = torch.zeros(max_len, embed_size)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pos_encoding', pos_encoding)

    def forward(self, x):
        x = x + self.pos_encoding[:x.size(0), :]
        return self.dropout(x)
    

class TStransformer(nn.Module):
    def __init__(self, conf: Configuration):
        super(TStransformer, self).__init__()
        
        self.dim_series = conf.getEntry("dim_series")
        self.len_series = conf.getEntry("len_series")
        self.len_reduce = conf.getEntry("len_reduce")
        self.embed_size = conf.getEntry("embed_size")
        self.num_heads = conf.getEntry("num_heads")
        self.num_layers = conf.getEntry("num_layers")
        
        self.embedding = nn.Linear(self.dim_series, self.embed_size)
        self.positional_encoding = PositionalEncoding(self.embed_size)
        self.transformer = nn.Transformer(self.embed_size, self.num_heads, self.num_layers)
        self.reduce = nn.Linear(self.len_series, self.len_reduce)
        self.de_embedding = nn.Linear(self.embed_size, self.dim_series)

    def forward(self, x):
        # x: (batch_size, len_series, dim_series)
        x = self.embedding(x)  # (batch_size, len_series, embed_size)
        x = self.positional_encoding(x.permute(1, 0, 2))  # (len_series, batch_size, embed_size)
        x = self.transformer(x, x)  # (len_series, batch_size, embed_size)
        x = x.permute(2, 1, 0)  # (embed_size, batch_size, len_series)
        x = self.reduce(x)  # (embed_size, batch_size, len_reduce)
        x = x.permute(1, 2, 0)  # (batch_size, len_reduce, embed_size)
        x = x.de_embedding(x)  # (batch_size, len_reduce, dim_series)
        return x
