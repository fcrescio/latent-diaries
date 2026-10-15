"""Sequence-aware continuous prefixes from numeric semantic factors.

This M1 probe consumes authored numeric inputs, not image features. No language
tokenizer, discrete text prediction, or per-example lookup is present.
"""

import torch
from torch import nn


class SequenceReader(nn.Module):
    def __init__(
        self, weather_dim, maximum_tokens, mean, scale, length_coef, width=128, layers=4, heads=4
    ):
        super().__init__()
        if maximum_tokens < 2 or weather_dim < 1 or heads < 1 or width < 1 or width % heads:
            raise ValueError("Invalid reader architecture")
        if mean.ndim != 1 or scale.shape != mean.shape or length_coef.shape != (weather_dim + 3,):
            raise ValueError("Normalization/length shapes differ")
        if not all(torch.isfinite(t).all() for t in [mean, scale, length_coef]) or not (scale > 0).all():
            raise ValueError("Finite normalization with positive scale required")
        self.weather_dim = weather_dim
        self.maximum_tokens = maximum_tokens
        self.width = width
        self.layers = layers
        self.heads = heads
        self.register_buffer("mean", mean.detach().clone())
        self.register_buffer("scale", scale.detach().clone())
        self.register_buffer("length_coef", length_coef.detach().clone())
        self.weather = nn.Linear(weather_dim, width)
        self.opening = nn.Linear(1, width)
        self.memory = nn.Linear(1, width)
        self.factor_types = nn.Parameter(torch.randn(1, 3, width) * 0.02)
        self.queries = nn.Parameter(torch.randn(1, maximum_tokens, width) * 0.02)
        self.relative_position = nn.Linear(2, width)
        layer = nn.TransformerDecoderLayer(
            width,
            heads,
            dim_feedforward=width * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, layers, norm=nn.LayerNorm(width))
        self.output = nn.Linear(width, mean.numel())

    def forward(self, features):
        if features.ndim != 2 or features.shape[1] != self.weather_dim + 2:
            raise ValueError("Expected numeric weather/opening/memory features")
        if not torch.is_floating_point(features) or not torch.isfinite(features).all():
            raise ValueError("Finite floating-point features required")
        ones = features.new_ones(features.shape[0], 1)
        lengths = torch.round(torch.cat([ones, features], -1) @ self.length_coef).long()
        lengths = lengths.clamp(1, self.maximum_tokens)
        position = torch.arange(self.maximum_tokens, device=features.device, dtype=features.dtype)
        coordinates = torch.stack(
            [
                (position / (self.maximum_tokens - 1)).expand(features.shape[0], -1),
                position[None] / (lengths[:, None] - 1).clamp_min(1),
            ],
            -1,
        )
        queries = self.queries + self.relative_position(coordinates)
        factors = (
            torch.stack(
                [
                    self.weather(features[:, : self.weather_dim]),
                    self.opening(features[:, self.weather_dim : self.weather_dim + 1]),
                    self.memory(features[:, self.weather_dim + 1 :]),
                ],
                1,
            )
            + self.factor_types
        )
        mask = position[None] < lengths[:, None]
        hidden = self.decoder(queries, factors, tgt_key_padding_mask=~mask)
        prefix = self.output(hidden) * self.scale + self.mean
        return prefix, mask, lengths


def masked_prefix_loss(predicted, target, mask, scale):
    if predicted.shape != target.shape or mask.shape != predicted.shape[:2]:
        raise ValueError("Prefix loss shapes differ")
    if mask.dtype != torch.bool or not mask.any():
        raise ValueError("Nonempty boolean mask required")
    error = ((predicted - target) / scale).square().mean(-1)
    return error[mask].mean()
