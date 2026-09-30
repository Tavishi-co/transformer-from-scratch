import torch
import torch.nn as nn

from .multi_head_attention import MultiHeadAttention
from .feed_forward import FeedForward


class TransformerDecoderLayer(nn.Module):

    def __init__(
        self,
        d_model=512,
        num_heads=8,
        d_ff=2048,
        dropout=0.1
    ):
        super().__init__()

        self.self_attn = MultiHeadAttention(
            d_model,
            num_heads
        )

        self.cross_attn = MultiHeadAttention(
            d_model,
            num_heads
        )

        self.ffn = FeedForward(
            d_model,
            d_ff
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x,
        encoder_output,
        causal=True
    ):
        # 1. Masked self-attention
        self_attn_output, _ = self.self_attn(
            x,
            x,
            x,
            causal=causal
        )

        x = self.norm1(
            x + self.dropout(self_attn_output)
        )

        # 2. Cross-attention
        cross_attn_output, _ = self.cross_attn(
            x,
            encoder_output,
            encoder_output,
            causal=False
        )

        x = self.norm2(
            x + self.dropout(cross_attn_output)
        )

        # 3. Feed-forward network
        ffn_output = self.ffn(x)

        x = self.norm3(
            x + self.dropout(ffn_output)
        )

        return x