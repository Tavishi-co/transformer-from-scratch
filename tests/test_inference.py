import torch

from src.transformer import Transformer


def test_generate_shape():

    model = Transformer(
        src_vocab_size=20,
        tgt_vocab_size=20,
        d_model=64,
        num_heads=4,
        num_layers=2,
        d_ff=256,
        max_len=20
    )

    src = torch.randint(
        3,
        20,
        (2, 6)
    )

    generated = model.generate(
        src,
        max_len=6,
        bos_token_id=1
    )

    assert generated.shape == (2, 6)