"""
Linear Centered Kernel Alignment (CKA) for comparing layer representations.

Reference: Kornblith et al., "Similarity of Neural Network Representations
           Revisited", ICML 2019.
"""

import torch


def centered_gram_linear(X):
    """Centered linear Gram matrix: H K H where H = I - 1/n."""
    K = X @ X.t()
    n = K.size(0)
    H = torch.eye(n, dtype=K.dtype, device=K.device) - 1.0 / n
    return H @ K @ H


def cka_linear(X, Y, eps=1e-12):
    """
    Linear CKA between two feature matrices.

    Parameters
    ----------
    X, Y : Tensor [N, D]
        Spatially-pooled features from N samples.

    Returns
    -------
    float
        CKA similarity in [0, 1].  Higher = more similar.
    """
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    Kx = centered_gram_linear(X)
    Ky = centered_gram_linear(Y)
    hsic = (Kx * Ky).sum()
    norm = torch.sqrt((Kx * Kx).sum() * (Ky * Ky).sum()) + eps
    return (hsic / norm).item()


def layerwise_cka(feats_ref, feats_cmp):
    """
    Compute CKA for each layer between two feature dictionaries.

    Parameters
    ----------
    feats_ref, feats_cmp : dict[str, Tensor[N, C]]

    Returns
    -------
    dict[str, float]
        Layer name → CKA score.
    """
    out = {}
    for lname in feats_ref:
        X = feats_ref[lname].float()
        Y = feats_cmp[lname].float()
        out[lname] = cka_linear(X, Y)
    return out
