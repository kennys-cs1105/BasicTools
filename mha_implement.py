"""
手撕 Multihead Attention
"""

import torch
import torch.functional as F
import math
from torch import nn


class multi_head_attention(nn.Module):
    def __init__(self, d_model, n_head):
        super(multi_head_attention, self).__init__()
        
        self.d_model = d_model
        self.n_head = n_head
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_combine = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, q, k, v):
        batch, time, dimension = q.shape
        n_d = self.d_model // self.n_head
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # dimension维度拆分 然后将time和self.n_head交换 
        q = q.view(batch, time, self.n_head, n_d).permute(0, 2, 1, 3)
        k = k.view(batch, time, self.n_head, n_d).permute(0, 2, 1, 3)
        v = v.view(batch, time, self.n_head, n_d).permute(0, 2, 1, 3)  
        
        score = q @ k.transpose(2, 3) / math.sqrt(n_d)
        mask = torch.tril(torch.ones(time, time, dtype=bool)) # torch.tril: 下三角矩阵
        score = score.masked_fill(mask == 0, float("-inf")) # 等于0的地方用负无穷表示
        score = self.softmax(score) @ v

        score = score.permute(0, 2, 1, 3).contiguous().view(batch, time, dimension)
        out = self.w_combine(score)

        return out


if __name__ == "__main__":

    X = torch.rand(128, 64, 512) # batch, time, dimension
    print(f"X shape is {X.shape}")

    # 设置MHA的基本参数
    d_model = 512 # QKV空间中的映射维度
    n_head = 8 # 头数
    
    mha = multi_head_attention(d_model, n_head)
    out = mha(X, X, X)
    print(f"out shape is {out.shape}")
