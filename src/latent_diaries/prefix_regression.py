"""Continuous factor regression output and numeric length prediction."""

import torch


def predict_prefix(features, prefix_coef, length_coef):
    if features.ndim != 1 or prefix_coef.ndim != 3:
        raise ValueError("Expected one numeric vector and [bias+factors,K,H] coefficients")
    if prefix_coef.shape[0] != features.numel() + 1 or length_coef.shape != prefix_coef.shape[:1]:
        raise ValueError("Feature/coefficient dimensions differ")
    if not all(
        torch.is_floating_point(t) and torch.isfinite(t).all()
        for t in [features, prefix_coef, length_coef]
    ):
        raise ValueError("Finite floating-point coefficients/features required")
    feature = torch.cat([features.new_ones(1), features])
    length = int(torch.round(feature @ length_coef).clamp(1, prefix_coef.shape[1]))
    prefix = torch.einsum("d,dkh->kh", feature, prefix_coef)[None, :length]
    return prefix, length
