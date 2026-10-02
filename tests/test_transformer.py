import torch

from src.transformer import Transformer


def test_transformer_shape():

    model = Transformer(
        src_vocab_size=1000,
        tgt_vocab_size=1200,
        d_model=128,
        num_heads=8,
        num_layers=2,
        d_ff=512
    )

    src = torch.randint(
        0,
        1000,
        (2, 10)
    )

    tgt = torch.randint(
        0,
        1200,
        (2, 12)
    )

    logits = model(src, tgt)

    assert logits.shape == (
        2,
        12,
        1200
    )