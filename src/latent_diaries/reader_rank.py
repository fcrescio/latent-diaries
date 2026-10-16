"""Training-only reconstruction bounds for readers with a linear output head."""
import torch


def reconstruction_rank_bound(prefixes, scale, width):
    """Bound full masked MSE using only positions valid in every training row.

Row centering removes the output bias. A width-W hidden representation followed
by a linear output can have centered rank at most W, irrespective of the input
network. Eckart-Young bounds the squared reconstruction error on the shared
valid block; dividing by ALL valid scalars makes it a bound on full masked MSE.
This does not bound speech quality or acoustic cross-entropy.
"""
    if not prefixes or width < 1 or scale.ndim != 1 or not (scale > 0).all():
        raise ValueError('Nonempty targets, positive width/scale required')
    if any(p.ndim != 2 or p.shape[1] != scale.numel() or not len(p)
           or not torch.isfinite(p).all() for p in prefixes) or not torch.isfinite(scale).all():
        raise ValueError('Finite compatible unpadded target matrices required')
    common = min(len(p) for p in prefixes)
    matrix = torch.stack([(p[:common].double() / scale.double()).flatten() for p in prefixes])
    matrix = matrix - matrix.mean(0)
    eigenvalues = torch.linalg.eigvalsh(matrix @ matrix.T).clamp_min(0).flip(0)
    denominator = sum(p.numel() for p in prefixes)
    return {'training_examples': len(prefixes), 'common_valid_positions': common,
            'decoder_width': width,
            'full_masked_normalized_mse_lower_bound': float(eigenvalues[width:].sum() / denominator),
            'scope': 'Numerical rank bound on training prefix reconstruction, not speech quality'}
