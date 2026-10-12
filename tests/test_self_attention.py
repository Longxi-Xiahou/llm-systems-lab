import torch

from src.llm_systems.attention.self_attention import SelfAttention


# -------------------------
# 1. Setting variables
# -------------------------

batch_size = 2
seq_len = 5
d_model = 8

d_q = 4
d_k = 4
d_v = 4


# -------------------------
# 2. Create input X
# -------------------------

x = torch.randn(
    batch_size,
    seq_len,
    d_model
)

print("Input X shape:", x.shape)


# -------------------------
# 3. Create Self-Attention
# -------------------------

self_attention = SelfAttention(
    d_model=d_model,
    d_q=d_q,
    d_k=d_k,
    d_v=d_v,
    causal=True
)


# -------------------------
# 4. Forward
# -------------------------

output, weights = self_attention(x)


# -------------------------
# 5. Check Output
# -------------------------

print("Output O shape:", output.shape)
print("Attention weights shape:", weights.shape)

print("\nAttention weights:")
print(weights)