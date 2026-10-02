import torch
import torch.nn as nn

from .encoder_layer import TransformerEncoderLayer
from .decoder_layer import TransformerDecoderLayer
from .positional_encoding import PositionalEncoding


class Transformer(nn.Module):

    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=512,
        num_heads=8,
        num_layers=6,
        d_ff=2048,
        max_len=5000,
        dropout=0.1
    ):
        super().__init__()

        self.d_model = d_model

        # Token embeddings
        self.src_embedding = nn.Embedding(
            src_vocab_size,
            d_model
        )

        self.tgt_embedding = nn.Embedding(
            tgt_vocab_size,
            d_model
        )

        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            d_model,
            max_len
        )

        # Encoder stack
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model,
                num_heads,
                d_ff,
                dropout
            )
            for _ in range(num_layers)
        ])

        # Decoder stack
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(
                d_model,
                num_heads,
                d_ff,
                dropout
            )
            for _ in range(num_layers)
        ])

        # Final vocabulary projection
        self.output_projection = nn.Linear(
            d_model,
            tgt_vocab_size
        )

        self.dropout = nn.Dropout(dropout)

    def encode(self, src):

        x = self.src_embedding(src)

        x = x * (self.d_model ** 0.5)

        x = self.positional_encoding(x)

        x = self.dropout(x)

        for layer in self.encoder_layers:
            x = layer(x)

        return x

    def decode(
        self,
        tgt,
        encoder_output
    ):

        x = self.tgt_embedding(tgt)

        x = x * (self.d_model ** 0.5)

        x = self.positional_encoding(x)

        x = self.dropout(x)

        for layer in self.decoder_layers:
            x = layer(
                x,
                encoder_output,
                causal=True
            )

        return x

    def forward(
        self,
        src,
        tgt
    ):

        encoder_output = self.encode(src)

        decoder_output = self.decode(
            tgt,
            encoder_output
        )

        logits = self.output_projection(
            decoder_output
        )

        return logits