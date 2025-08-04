"""
实现transformer
"""

import torch
import torch.nn as nn
import math
from encoder_embedding import multi_head_attention, PositionWiseFeedForward, LayerNormalization
from decoder_layer import DecoderLayer
from encoder_embedding import EncoderEmbedding, DecoderEmbedding
from encoder_decoder import Encoder, Decoder


class Transformer(nn.Module):
    def __init__(self, src_pad_idx, tgt_pad_idx, enc_vocab_size, dec_vocab_size, max_len, d_model, n_heads, hidden, drop_prob, n_layers, device):
        super(Transformer).__init__()
        
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.device = device
        
        self.encoder = Encoder(enc_vocab_size, max_len, d_model, n_heads, hidden, drop_prob, n_layers, device)
        self.decoder = Decoder(dec_vocab_size, max_len, d_model, n_heads, hidden, drop_prob, n_layers, device)

    def make_casual_mask(self, q, k):
        """
        q: [batch_size, q_len]
        k: [batch_size, k_len]
        """
        len_q, len_k = q.size(1), k.size(1)
        mask = torch.tril(torch.ones(len_q, len_k)).type(torch.BoolTensor).to(self.device) # 下三角掩码
        return mask
    
    def make_pad_mask(self, q, k, pad_idx_q, pad_idx_k):
        """
        (batch, time, len_q, len_k)
        """
        len_q, len_k = q.size(1), k.size(1)
        q = q.ne(pad_idx_q).unsqueeze(1).unsqueeze(3) # [batch_size, 1, 1, q_len]
        q = q.repeat(1, 1, 1, len_k)

        k = k.ne(pad_idx_k).unsqueeze(1).unsqueeze(2) # [batch_size, 1, k_len, 1]
        k = k.repeat(1, 1, len_q, 1)

        mask = q & k 
        return mask

    def forward(self, src, tgt):
        """
        src: [batch_size, src_len]
        tgt: [batch_size, tgt_len]
        """
        src_mask = self.make_pad_mask(src, src, self.src_pad_idx, self.src_pad_idx) # encoder中的pad mask
        tgt_mask = self.make_pad_mask(tgt, tgt, self.tgt_pad_idx, self.tgt_pad_idx) # decoder中的因果mask
        casual_mask = self.make_casual_mask(tgt, tgt)
        tgt_mask = tgt_mask * casual_mask
        src_tgt_mask = self.make_pad_mask(tgt, src, self.tgt_pad_idx, self.src_pad_idx)

        enc_output = self.encoder(src, src_mask) # [batch_size, src_len, d_model]
        dec_output = self.decoder(tgt, enc_output, tgt_mask, src_tgt_mask) # [batch_size, tgt_len, d_model]
        
        return dec_output

