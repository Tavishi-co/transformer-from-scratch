import torch

from src.attention import scaled_dot_product_attention


def test_attention_shape():
    Q = torch.randn(2, 4, 8)
    K = torch.randn(2, 4, 8)
    V = torch.randn(2, 4, 8)

    output, weights = scaled_dot_product_attention(Q, K, V)

    assert output.shape == (2, 4, 8)
    assert weights.shape == (2, 4, 4)


def test_attention_weights_sum_to_one():
    Q = torch.randn(2, 4, 8)
    K = torch.randn(2, 4, 8)
    V = torch.randn(2, 4, 8)

    _, weights = scaled_dot_product_attention(Q, K, V)

    row_sums = weights.sum(dim=-1)

    assert torch.allclose(
        row_sums,
        torch.ones_like(row_sums),
        atol=1e-6
    )


def test_causal_attention():
    Q = torch.randn(1, 4, 8)
    K = torch.randn(1, 4, 8)
    V = torch.randn(1, 4, 8)

    _, weights = scaled_dot_product_attention(
        Q, K, V, causal=True
    )

    upper_triangle = torch.triu(
        weights,
        diagonal=1
    )

    assert torch.all(upper_triangle == 0)