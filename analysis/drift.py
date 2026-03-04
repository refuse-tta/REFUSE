"""
Feature drift analysis — measures how much layer-wise statistics
shift between a reference (e.g. base-clean) and a comparison
(e.g. adapted-corrupt) model.
"""

import torch


def mean_std_stats(X):
    """Channel-wise mean and std from [N, C] features."""
    return X.mean(dim=0), X.std(dim=0, unbiased=False)


def drift_score(mu_a, std_a, mu_b, std_b, eps=1e-8):
    """
    Relative mean shift and relative std shift between two distributions.

    Returns (rel_mean_drift, rel_std_drift).
    """
    d_mu = (mu_a - mu_b).abs().mean().item()
    d_std = (std_a - std_b).abs().mean().item()
    norm_mu = mu_b.abs().mean().item() + eps
    norm_std = std_b.abs().mean().item() + eps
    return d_mu / norm_mu, d_std / norm_std


def compute_layerwise_drift(feats_ref, feats_cmp):
    """
    Compute per-layer drift between reference and comparison features.

    Parameters
    ----------
    feats_ref : dict[str, Tensor[N, C]]
        Reference features (e.g. base model on clean data).
    feats_cmp : dict[str, Tensor[N, C]]
        Comparison features (e.g. adapted model on corrupt data).

    Returns
    -------
    dict[str, tuple[float, float]]
        Layer name → (relative_mean_drift, relative_std_drift).
        Lower is better (closer to the clean-data regime).
    """
    out = {}
    for lname in feats_ref:
        mu_r, std_r = mean_std_stats(feats_ref[lname])
        mu_c, std_c = mean_std_stats(feats_cmp[lname])
        out[lname] = drift_score(mu_c, std_c, mu_r, std_r)
    return out
