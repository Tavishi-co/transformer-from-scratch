import torch
import torch.nn as nn

from .attention import scaled_dot_product_attention


class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        d_model=512,
        num_heads=8
    ):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        query,
        key,
        value,
        causal=False
    ):
        B, T_q, _ = query.shape
        T_k = key.shape[1]

        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)

        Q = Q.view(
            B,
            T_q,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        K = K.view(
            B,
            T_k,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        V = V.view(
            B,
            T_k,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        output, weights = scaled_dot_product_attention(
            Q,
            K,
            V,
            causal=causal
        )

        output = output.transpose(1, 2).contiguous()

        output = output.view(
            B,
            T_q,
            self.d_model
        )

        output = self.out_proj(output)

        return output, weights