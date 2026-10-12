from .qkv_projection import QKVProjection
from .attention import ScaledDotProductAttention
import torch
import torch.nn as nn

class SelfAttention(nn.Module):

    def __init__(self, d_model: int, d_q: int, d_k: int, d_v: int, bias: bool = False, causal=False):
        super().__init__()
        self.projection = QKVProjection(d_model, d_q, d_k, d_v, bias=bias)
        self.attention = ScaledDotProductAttention(causal=causal)


    def forward(self, x: torch.Tensor):
        Q, K, V = self.projection(x)
        O, W = self.attention(Q, K, V, padding_mask = None)

        return O, W
