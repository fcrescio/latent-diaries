"""Small continuous-prefix bridge with separate permanent image and last audio.

No strings, linguistic IDs, target tensors or example lookup are accepted by
forward. Image-only and feedback controls have exactly the same parameters.
"""

import torch
from torch import nn


class TemporalReader(nn.Module):
    def __init__(self, maximum_tokens, mean, scale, *, feedback=True, width=128):
        super().__init__()
        if maximum_tokens < 1 or width < 1 or mean.shape != (1024,) or scale.shape != mean.shape:
            raise ValueError("Expected positive capacity and1024D normalization")
        if not torch.isfinite(mean).all() or not torch.isfinite(scale).all() or not (scale > 0).all():
            raise ValueError("Finite mean and positive scale required")
        self.maximum_tokens = maximum_tokens
        self.feedback = feedback
        self.width = width
        self.register_buffer("mean", mean.detach().clone())
        self.register_buffer("scale", scale.detach().clone())
        self.image_norm = nn.LayerNorm(768)
        self.audio_norm = nn.LayerNorm(768)
        self.hidden = nn.Sequential(nn.Linear(1537, width), nn.GELU())
        self.output = nn.Linear(width, maximum_tokens * 1024)
        self.length = nn.Linear(width, 1)

    def forward(self, image, last_audio, present):
        if image.ndim != 2 or image.shape[1] != 768 or last_audio.shape != image.shape:
            raise ValueError("Expected separate image and last-audio [B,768] tensors")
        if present.shape != (image.shape[0],) or present.dtype != torch.bool:
            raise ValueError("Explicit boolean history-presence mask required")
        if image.device != last_audio.device or image.device != present.device:
            raise ValueError("Inputs must share a device")
        if not all(torch.is_floating_point(x) and torch.isfinite(x).all() for x in [image, last_audio]):
            raise ValueError("Finite floating-point features required")
        # Mask AFTER normalization: even a learned LayerNorm bias cannot leak a
        # supplied history into absent-history or image-only inference.
        gate = present[:, None].to(image.dtype) * int(self.feedback)
        hidden = self.hidden(
            torch.cat([self.image_norm(image), self.audio_norm(last_audio) * gate, gate], -1)
        )
        prefix = self.output(hidden).reshape(-1, self.maximum_tokens, 1024) * self.scale + self.mean
        continuous_length = self.length(hidden)[:, 0]
        lengths = continuous_length.round().long().clamp(1, self.maximum_tokens)
        mask = torch.arange(self.maximum_tokens, device=image.device)[None] < lengths[:, None]
        return prefix, mask, lengths, continuous_length


def prefix_objective(predicted, target, mask, scale, length_prediction, lengths):
    """Training-only masked projection regression and a learned length head."""
    if predicted.shape != target.shape or mask.shape != target.shape[:2] or mask.dtype != torch.bool:
        raise ValueError("Aligned prefix target and boolean token mask required")
    if not mask.any() or not torch.equal(mask.sum(-1), lengths):
        raise ValueError("Lengths must match nonempty gold masks")
    error = ((predicted - target.detach()) / scale.detach()).square().mean(-1)
    reconstruction = error[mask].mean()
    length_loss = ((length_prediction - lengths) / target.shape[1]).square().mean()
    return reconstruction + length_loss, reconstruction, length_loss
