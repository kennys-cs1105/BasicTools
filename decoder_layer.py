"""
decoder layer
1. mask attention
2. cross attention
"""

import torch
import torch.nn as nn
import math
from encoder_embedding import multi_head_attention, PositionWiseFeedForward, LayerNormalization


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, hidden, drop_prob):
        super().__init__()
        self.attention1 = multi_head_attention(d_model, n_heads)
        self.cross_attention = multi_head_attention(d_model, n_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, hidden, drop_prob)
        self.layer_norm1 = LayerNormalization(d_model)
        self.layer_norm2 = LayerNormalization(d_model)
        self.layer_norm3 = LayerNormalization(d_model)
        self.dropout1 = nn.Dropout(drop_prob)
        self.dropout2 = nn.Dropout(drop_prob)
        self.dropout3 = nn.Dropout(drop_prob)
        self.ffn = PositionWiseFeedForward(d_model, hidden, drop_prob)

    def forward(self, dec, enc, t_mask, s_mask):
        """
        dec: [batch_size, tgt_len, d_model]
        enc: [batch_size, src_len, d_model]
        t_mask: [batch_size, 1, tgt_len, tgt_len]
        s_mask: [batch_size, 1, 1, src_len]
        """
        _x = dec
        x = self.attention1(dec, dec, dec, t_mask) # 下三角掩码 带mask的掩码

        x = self.dropout1(x)
        x = self.layer_norm1(x + _x)

        if enc is not None:
            _x = x
            x = self.cross_attention(x, enc, enc, s_mask) # 未知掩码
            x = self.dropout2(x)
            x = self.layer_norm2(x + _x)

        _x = x
        x = self.ffn(x)
        x = self.dropout3(x)
        x = self.layer_norm3(x + _x)

        return x