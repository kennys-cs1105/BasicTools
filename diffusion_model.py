import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

"""
扩散模型
1. 位置编码
2. 前向过程 MLP
"""

class SinusoidalPosEmb(nn.Module):
    """
    Diffusion Model中的位置编码
    """
    def __init__(self, dim):
        super(SinusoidalPosEmb, self).__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class MLP(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, device, t_dim):
        super(MLP, self).__init__()
        
        self.t_dim = t_dim
        self.device = device
        self.a_dim = action_dim

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim*2),
            nn.Mish(),
            nn.Linear(t_dim*2, t_dim)
        )

        input_dim = state_dim + action_dim + t_dim
        
        self.mid_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, action_dim),
            nn.Mish()
        )

        self.final_layer = nn.Linear(hidden_dim, action_dim)

        self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, time, state):
        t_emb = self.time_mlp(time)
        x = torch.cat([x, t_emb, state], dim=1)
        x = self.mid_layer(x)
        x = self.final_layer(x)
        return x


class DiffusionModel(nn.Module):
    def __init__(self, loss_type, beta_schedule="linear", clip_denoised=True, **kwargs):
        super(DiffusionModel, self).__init__()
        
        self.state_dim = kwargs["state_dim"]
        self.action_dim = kwargs["action_dim"]
        self.hidden_dim = kwargs["hidden_dim"]
        self.t_dim = kwargs["t_dim"]
        self.T = kwargs["T"]
        self.device = torch.device(kwargs["device"])

        if beta_schedule == "linear":
            betas = torch.linspace(0.0001, 0.02, self.T, dtype=torch.float32)
        
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0) # [1,2,3] -> [1,1*2,1*2*3]
        alphas_cumprod_prev = torch.cat([torch.ones(1, dtype=torch.float32), alphas_cumprod[:-1]])
          
