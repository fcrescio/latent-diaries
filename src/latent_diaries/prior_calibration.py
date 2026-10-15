"""Train numeric image calibration through a frozen continuous language prior."""
import torch
from torch import nn
from torch.nn import functional as F


class NumericImageAdapter(nn.Module):
    def __init__(self, coefficient):
        super().__init__()
        if coefficient.shape != (769, 768) or not torch.isfinite(coefficient).all():
            raise ValueError('Finite affine image calibration required')
        self.coefficient = nn.Parameter(coefficient.clone())

    def forward(self, features):
        return F.normalize(torch.cat([features.new_ones(len(features), 1), features], -1) @ self.coefficient, dim=-1)


def opening_nll(prior, semantic, prefix, lengths, temperature=0.01):
    """Shift gold vectors; score only valid opening positions, never future inputs."""
    if temperature <= 0 or prefix.ndim != 3 or lengths.shape != (len(prefix),):
        raise ValueError('Positive temperature and aligned continuous openings required')
    if not bool(((lengths > 0) & (lengths <= prefix.shape[1])).all()):
        raise ValueError('Opening lengths outside cached vectors')
    gold = prior.encode(prefix)
    previous = torch.cat([torch.zeros_like(gold[:, :1]), gold[:, :-1]], 1)
    # cuDNN's eval GRU does not retain the reserve needed for input gradients.
    # The native path keeps frozen-reader semantics while differentiating inputs.
    with torch.backends.cudnn.flags(enabled=False):
        context = prior.contexts(semantic, previous)
    means, routing = prior.predictions(context)
    error = ((means - gold[:, :, None]) * prior.latent_scale).square().sum(-1) / prior.mean.numel()
    nll = -torch.logsumexp(F.log_softmax(routing, -1) - error / temperature, -1)
    positions = torch.arange(prefix.shape[1], device=prefix.device)[None]
    weight = (positions < lengths[:, None]) * torch.where(positions < 8, 2.0, 1.0)
    return (nll * weight).sum(-1) / weight.sum(-1)


def shared_opening_length(prefixes, lengths):
    if len(prefixes) < 2:
        raise ValueError('Opening boundary requires multiple training continuations')
    same = (prefixes == prefixes[:1]).all(-1).all(0)
    different = (~same).nonzero()
    boundary = int(different[0, 0]) if len(different) else prefixes.shape[1]
    boundary = min(boundary, int(lengths.min()))
    if boundary < 2:
        raise ValueError('No common authored opening')
    return boundary
