"""
Transormer Encoder and Decoder
1. Encoder
2. Decoder
"""


import torch
import torch.nn as nn
import math
from encoder_embedding import multi_head_attention, PositionWiseFeedForward, LayerNormalization, TransformerEmbedding, EncoderLayer
from decoder_layer import DecoderLayer


class Encoder(nn.Module):
    def __init__(self, enc_vocab_size, max_len, d_model, n_heads, hidden, drop_prob, n_layers, device):
        super(Encoder).__init__()
        
        self.embedding = TransformerEmbedding(d_model=d_model, max_len=max_len, vocab_size=enc_vocab_size, drop_prob=drop_prob, device=device)

        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, hidden, drop_prob) for _ in range(n_layers)
        ])

    def forward(self, x, s_mask):
        """
        x: [batch_size, src_len]
        s_mask: [batch_size, 1, 1, src_len]
        """
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, s_mask)
        return x
    

class Decoder(nn.Module):
    def __init__(self, dec_vocab_size, max_len, d_model, n_heads, hidden, drop_prob, n_layers, device):
        super(Decoder).__init__()
        self.embedding = TransformerEmbedding(d_model=d_model, max_len=max_len, vocab_size=dec_vocab_size, drop_prob=drop_prob, device=device)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, hidden, drop_prob) for _ in range(n_layers)
        ])
        self.fc = nn.Linear(d_model, dec_vocab_size)

    def forward(self, dec, enc, t_mask, s_mask):
        """
        dec: [batch_size, tgt_len]
        t_mask: [batch_size, 1, tgt_len, tgt_len]
        s_mask: [batch_size, 1, 1, src_len]
        enc: [batch_size, src_len, d_model]
        """
        dec = self.embedding(dec)
        for layer in self.layers:
            dec = layer(dec, enc, t_mask, s_mask)
        dec = self.fc(dec)
        return dec
