import math 
import torch as T
import torch.nn as nn 

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=200):
        super(PositionalEncoding, self).__init__()

        pe = T.zeros(max_len, d_model)
        position = T.arange(0, max_len, dtype=T.float).unsqueeze(1)
        div_term = T.exp(T.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = T.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = T.cos(position * div_term)
        else:
            pe[:, 1::2] = T.cos(position * div_term[:-1])

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]