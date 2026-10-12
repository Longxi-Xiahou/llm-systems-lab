import torch
import torch.nn as nn


class QKVProjection(nn.Module):

    def __init__(self, d_model: int, d_q: int, d_k: int, d_v: int, bias: bool = False):
        super().__init__()
        self.W_Q = nn.Linear(d_model, d_q, bias=bias)
        self.W_K = nn.Linear(d_model, d_k, bias=bias)
        self.W_V = nn.Linear(d_model, d_v, bias=bias)

    def forward(self, x: torch.Tensor):
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        return Q, K, V