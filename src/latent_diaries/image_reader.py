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


def validate_image_inputs(rows, known_ids, allow_new=False):
    """Validate image-vector-only inference, including genuinely new image IDs."""
    import math

    seen = set()
    for row in rows:
        if set(row) != {'id', 'split', 'features'}:
            raise ValueError('Image-vector-only evaluation spec required')
        label = row['id']
        if not isinstance(label, str) or not label or not all(c.isalnum() or c in '-_' for c in label):
            raise ValueError('Unsafe image ID')
        if label in seen:
            raise ValueError('Duplicate image ID')
        seen.add(label)
        if row['split'] not in {'train', 'heldout'}:
            raise ValueError('Unknown evaluation split')
        if label not in known_ids and not allow_new:
            raise ValueError('Unknown ID; explicitly enable new-image inference')
        if (not isinstance(row['features'], list) or len(row['features']) != 768 or
                any(type(v) not in {int, float} or not math.isfinite(v) for v in row['features'])):
            raise ValueError('Expected finite 768D image vector')
