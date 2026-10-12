import torch
import torch.nn as nn

class ScaledDotProductAttention(nn.Module):

    def __init__(self, causal=False):
        super().__init__()
        self.causal = causal


    def forward(self, query, key, value, padding_mask=None):
        l_q = query.shape[-2]
        d_q = query.shape[-1]
        l_k = key.shape[-2]
        d_k = key.shape[-1]
        l_v = value.shape[-2]

        if d_q != d_k:
            raise ValueError("Invalid dimension")

        if l_k != l_v:
            raise ValueError("Key and Value sequence lengths must match")

        if query.shape[0] != key.shape[0] or key.shape[0] != value.shape[0]:
            raise ValueError("Batch sizes must match")

        scores = query @ key.transpose(-2, -1)

        scores = scores / (d_k ** 0.5)

        if self.causal:
            mask = torch.triu(
                torch.ones(l_q, l_k, device=scores.device, dtype=torch.bool),   # ensure mask on same device | data type = boolean: true
                diagonal=1
            )   # .bool()

            scores = scores.masked_fill(mask, float("-inf"))    # where mask is True, replace the score with negative infinity.

        if padding_mask is not None:
            padding_mask = padding_mask.unsqueeze(1)    # (B, 1, Lk)
            scores = scores.masked_fill(padding_mask, float("-inf"))

        weights = torch.softmax(scores, dim=-1)

        output = weights @ value

        return output, weights


if __name__ == "__main__":
    query = torch.randn(3, 3, 4)
    key = torch.randn(3, 2, 4)
    value = torch.randn(3, 2, 5)

    attention = ScaledDotProductAttention(causal=True)
    output, weights = attention(query, key, value, padding_mask=None)

    print(output)
    print("Output shape:", output.shape)
    print(weights)
    print("Output weights:", weights.shape)

    print(weights[0])