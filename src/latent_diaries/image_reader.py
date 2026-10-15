"""Global image-vector baseline: small MLP to continuous Qwen prefix vectors."""
import torch
from torch import nn


class GlobalImageReader(nn.Module):
    def __init__(self, maximum_tokens, mean, scale, length_coef, width=128):
        super().__init__()
        if maximum_tokens < 1 or width < 1 or mean.shape != (1024,) or scale.shape != mean.shape:
            raise ValueError("Invalid image reader dimensions")
        if length_coef.shape != (769,):
            raise ValueError("Expected bias plus 768D length coefficients")
        if not all(torch.isfinite(t).all() for t in [mean, scale, length_coef]) or not (scale > 0).all():
            raise ValueError("Finite buffers with positive scale required")
        self.maximum_tokens = maximum_tokens
        self.register_buffer("mean", mean.detach().clone())
        self.register_buffer("scale", scale.detach().clone())
        self.register_buffer("length_coef", length_coef.detach().clone())
        self.network = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, width), nn.GELU(),
                                     nn.Linear(width, maximum_tokens * 1024))

    def forward(self, features):
        if features.ndim != 2 or features.shape[1] != 768:
            raise ValueError("Expected [B,768] global image vectors")
        if not torch.is_floating_point(features) or not torch.isfinite(features).all():
            raise ValueError("Finite floating-point image vectors required")
        design = torch.cat([features.new_ones(features.shape[0], 1), features], dim=-1)
        lengths = torch.round(design @ self.length_coef).long().clamp(1, self.maximum_tokens)
        mask = torch.arange(self.maximum_tokens, device=features.device)[None] < lengths[:, None]
        prefix = self.network(features).reshape(-1, self.maximum_tokens, 1024)
        return prefix * self.scale + self.mean, mask, lengths
