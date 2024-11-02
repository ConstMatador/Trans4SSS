from torch import nn


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


class TransformerModel(nn.Module):
    def __init__(self, num_tokens, embed_size, num_heads, num_layers):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(num_tokens, embed_size)
        self.positional_encoding = PositionalEncoding(embed_size)
        self.transformer = nn.Transformer(embed_size, num_heads, num_layers)
        self.fc = nn.Linear(embed_size, num_tokens)

    def forward(self, x):
        # x: (batch_size, seq_len)
        x = self.embedding(x)  # (batch_size, seq_len, embed_size)
        x = self.positional_encoding(x.permute(1, 0, 2))  # (seq_len, batch_size, embed_size)
        x = self.transformer(x)  # (seq_len, batch_size, embed_size)
        x = x.permute(1, 0, 2)  # (batch_size, seq_len, embed_size)
        x = self.fc(x)  # (batch_size, seq_len, num_tokens)
        return x