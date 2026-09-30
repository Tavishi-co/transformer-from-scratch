import torch

from src.decoder_layer import TransformerDecoderLayer


def test_decoder_layer_shape():

    batch_size = 2
    target_length = 10
    source_length = 12
    d_model = 512

    decoder = TransformerDecoderLayer(
        d_model=d_model,
        num_heads=8
    )

    x = torch.randn(
        batch_size,
        target_length,
        d_model
    )

    encoder_output = torch.randn(
        batch_size,
        source_length,
        d_model
    )

    output = decoder(
        x,
        encoder_output
    )

    assert output.shape == (
        batch_size,
        target_length,
        d_model
    )