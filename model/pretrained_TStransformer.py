import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from utils.conf import Configuration

class PositionalEncoding(nn.Module):
    def __init__(self, conf: Configuration, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.max_len = conf.getEntry('len_series')
        self.embed_size = conf.getEntry('embed_size')
        self.device = conf.getEntry('device')

        position = torch.arange(self.max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embed_size, 2).float() * -(torch.log(torch.tensor(10000.0)) / self.embed_size))
        pos_encoding = torch.zeros(self.max_len, self.embed_size).to(self.device)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pos_encoding', pos_encoding)

    def forward(self, x):
        x = x + self.pos_encoding[:x.size(1), :]
        return self.dropout(x)


class PretrainedTStransformer(nn.Module):
    def __init__(self, conf: Configuration, pretrained_model_name="bert-base-uncased"):
        super(PretrainedTStransformer, self).__init__()

        self.conf = conf
        self.dim_series = conf.getEntry("dim_series")
        self.len_series = conf.getEntry("len_series")
        self.len_reduce = conf.getEntry("len_reduce")
        self.embed_size = conf.getEntry("embed_size")
        self.num_heads = conf.getEntry("num_heads")
        self.num_layers = conf.getEntry("num_layers")
        self.batch_size = conf.getEntry("batch_size")

        # 加载预训练模型
        self.pretrained_model = AutoModel.from_pretrained(pretrained_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

        # 使用 PositionalEncoding
        self.positional_encoding = PositionalEncoding(self.conf)

        # 线性层将序列长度压缩为目标长度
        self.reduce = nn.Linear(self.len_series, self.len_reduce)

        # 解码器层，将模型输出映射回到原始特征空间
        self.de_embedding = nn.Linear(self.embed_size, self.dim_series)

    def forward(self, x):
        # x: (batch_size, dim_series, len_series)
        
        # 转置输入，将序列长度移到前面
        x = x.permute(0, 2, 1)  # (batch_size, len_series, dim_series)
        
        # 进行嵌入操作，将特征映射到嵌入空间
        x = self.tokenizer(x, return_tensors="pt", padding=True, truncation=True)
        inputs = x["input_ids"].to(self.pretrained_model.device)
        
        # 获取预训练模型的输出（例如BERT的输出）
        model_output = self.pretrained_model(inputs).last_hidden_state  # (batch_size, len_series, embed_size)
        
        # 使用位置编码
        x = self.positional_encoding(model_output)  # (batch_size, len_series, embed_size)
        
        # 压缩序列长度
        x = self.reduce(x)  # (batch_size, len_reduce, embed_size)

        # 映射回原始特征空间
        x = self.de_embedding(x)  # (batch_size, len_reduce, dim_series)
        
        return x
