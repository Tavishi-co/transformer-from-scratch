import math
import torch
import torch.nn.functional as F


def scaled_dot_product_attention(Q, K, V, causal=False):
    """
    Compute scaled dot-product attention.

    Args:
        Q: Query tensor (..., Tq, d_k)
        K: Key tensor   (..., Tk, d_k)
        V: Value tensor (..., Tk, d_v)
        causal: Whether to prevent attending to future tokens.

    Returns:
        output: (..., Tq, d_v)
        weights: (..., Tq, Tk)
    """

    d_k = Q.size(-1)

    
    scores = Q @ K.transpose(-2, -1)

    
    scores = scores / math.sqrt(d_k)

    
    if causal:
        Tq = scores.size(-2)
        Tk = scores.size(-1)

        mask = torch.triu(
            torch.ones(
                Tq,
                Tk,
                device=scores.device,
                dtype=torch.bool
            ),
            diagonal=1
        )

        scores = scores.masked_fill(mask, float("-inf"))

    
    weights = F.softmax(scores, dim=-1)

    
    output = weights @ V

    return output, weights