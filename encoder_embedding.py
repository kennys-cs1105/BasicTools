"""
手写token和position layer
手写编码模块 构建encoder layer
"""

import torch
import torch.functional as F
import math
from torch import nn

from mha_implement import multi_head_attention


class TokenEmbedding(nn.Embedding):
    """
    TokenEmbedding
    """
    def __init__(self, d_model, vocab_size):
        super(TokenEmbedding, self).__init__(d_model, vocab_size, padding_idx=1)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model


class PositionalEmbedding(nn.Module):
    """
    PositionalEmbedding
    """
    def __init__(self, d_model, max_len, device):
        super(PositionalEmbedding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False # 不需要梯度更新

        pos = torch.arange(0, max_len, dtype=torch.float, device=device)
        pos = pos.unsqueeze(1)
        _2i = torch.arange(0, d_model, 2, dtype=torch.float, device=device)
        
        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

    def forward(self, x):
        seq_len = x.shape[1]
        return self.encoding[:seq_len, :]


class LayerNormalization(nn.Module):
    """
    LayerNormalization
    """
    def __init__(self, d_model, eps=1e-10):
        super(LayerNormalization, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiasd=False)
        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out


class PositionWiseFeedForward(nn.Module):
    """
    PositionWiseFeedForward
    """
    def __init__(self, d_model, hidden, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransformerEmbedding(nn.Module):
    """
    TransformerEmbedding
    """
    def __init__(self, d_model, vocab_size, max_len, device, drop_prob):
        super(TransformerEmbedding, self).__init__()
        self.token_embedding = TokenEmbedding(d_model, vocab_size)
        self.position_embedding = PositionalEmbedding(d_model, max_len, device)
        self.layer_norm = LayerNormalization(d_model)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x):
        token_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(x)
        return self.dropout(token_emb + pos_emb)
    

class EncoderLayer(nn.Module):
    """
    EncoderLayer
    """
    def __init__(self, d_model, n_head, hidden, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = multi_head_attention(d_model, n_head)
        self.ffn = PositionWiseFeedForward(d_model, hidden, drop_prob)
        self.layer_norm1 = LayerNormalization(d_model)
        self.layer_norm2 = LayerNormalization(d_model)
        self.dropout1 = nn.Dropout(drop_prob)
        self.dropout2 = nn.Dropout(drop_prob)

    def forward(self, x, mask=None):
        _x = x
        x = self.attention(x, x, x, mask)

        x = self.dropout1(x)
        x = self.layer_norm1(x + _x)

        _x = x
        x = self.ffn(x)

        x = self.dropout2(x)
        x = self.layer_norm2(x + _x)

        return x
