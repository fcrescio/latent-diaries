"""Image-conditioned residuals around a fixed learned continuous speech prefix."""
import hashlib

import torch

from .image_reader import GlobalImageReader


def make_anchored_reader(anchor, width=256):
    if anchor.ndim != 2 or anchor.shape[1] != 1024 or not len(anchor) or not torch.isfinite(anchor).all():
        raise ValueError('Expected finite [K,1024] numeric language anchor')
    mean = anchor.mean(0)
    scale = anchor.std(0, unbiased=False).clamp_min(1e-4)
    lengths = anchor.new_zeros(769)
    lengths[0] = len(anchor)
    reader = GlobalImageReader(len(anchor), mean, scale, lengths, width=width)
    with torch.no_grad():
        reader.network[-1].weight.zero_()
        reader.network[-1].bias.copy_(((anchor - mean) / scale).flatten())
    return reader


def initialization_fingerprint(reader):
    value = hashlib.sha256()
    for name, tensor in sorted(reader.state_dict().items()):
        value.update(name.encode())
        value.update(str((tensor.dtype, tuple(tensor.shape))).encode())
        value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()
