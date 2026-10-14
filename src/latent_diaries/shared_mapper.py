"""Shared numeric-latent probe. No tokenizer, captions or per-example parameters."""

import torch
from torch import nn


class SharedMapper(nn.Module):
    def __init__(self, base, input_dim=3, width=128):
        super().__init__()
        if base.ndim != 3 or base.shape[0] != 1:
            raise ValueError("Base must be [1, tokens, hidden]")
        self.register_buffer("base", base.detach().clone())
        self.input_dim = input_dim
        self.width = width
        self.net = nn.Sequential(nn.Linear(input_dim, width), nn.Tanh(), nn.Linear(width, base.numel()))
        nn.init.normal_(self.net[-1].weight, std=0.001)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features):
        if features.ndim != 2 or features.shape[1] != self.input_dim:
            raise ValueError("Expected [batch, numeric factors]")
        if not torch.is_floating_point(features) or not torch.isfinite(features).all():
            raise ValueError("Features must be finite floating-point numbers")
        return self.base + self.net(features).reshape(-1, *self.base.shape[1:])


def validate_split(rows):
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)) or not rows:
        raise ValueError("Example IDs must be nonempty and unique")
    train = [r for r in rows if r["split"] == "train"]
    heldout = [r for r in rows if r["split"] == "heldout"]
    if len(train) + len(heldout) != len(rows) or not train or not heldout:
        raise ValueError("Require nonempty train and heldout splits")
    keys = [tuple(r["features"]) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate features cross the experimental split")
    return train, heldout
