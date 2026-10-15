"""Continuous latent prefix distribution; no tokenizer, text IDs or target lookup."""
import math

import torch
from torch import nn


class PrefixFlow(nn.Module):
    def __init__(self, mean, scale, center, basis, latent_scale, width=1024):
        super().__init__()
        if basis.ndim != 2 or basis.shape[1] != center.numel() or center.numel() % mean.numel():
            raise ValueError('Invalid continuous prefix subspace')
        if scale.shape != mean.shape or latent_scale.shape != (len(basis),):
            raise ValueError('Normalization shapes differ')
        if not all(torch.isfinite(v).all() for v in [mean, scale, center, basis, latent_scale]):
            raise ValueError('Finite numeric buffers required')
        if not bool((scale > 0).all() and (latent_scale > 0).all()):
            raise ValueError('Positive normalization required')
        self.tokens, self.hidden = center.numel() // mean.numel(), mean.numel()
        self.dimension = len(basis)
        for key, value in [('mean',mean),('scale',scale),('center',center),
                           ('basis',basis),('latent_scale',latent_scale)]:
            self.register_buffer(key,value.clone())
        self.feature = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, width))
        self.time = nn.Sequential(nn.Linear(16, width), nn.SiLU(), nn.Linear(width, width))
        self.input = nn.Linear(self.dimension + 1, width)
        self.network = nn.Sequential(nn.SiLU(), nn.Linear(width, width), nn.SiLU(),
                                     nn.Linear(width, width), nn.SiLU())
        self.output = nn.Linear(width, self.dimension + 1)

    def velocity(self, state, features, time):
        if features.ndim != 2 or features.shape[1] != 768 or state.shape != (len(features), self.dimension + 1):
            raise ValueError('Expected numeric state and [B,768] semantic/image vectors')
        if time.shape != (len(features),):
            raise ValueError('Expected one continuous time per sample')
        if not all(torch.isfinite(v).all() for v in [state, features, time]):
            raise ValueError('Finite numeric flow inputs required')
        frequencies = 2 ** torch.arange(8, device=time.device, dtype=time.dtype)
        phases = time[:, None] * frequencies * math.pi
        temporal = self.time(torch.cat([phases.sin(), phases.cos()], -1))
        return self.output(self.network(self.input(state) + self.feature(features) + temporal))

    def pack(self, prefix, lengths):
        if prefix.shape[1:] != (self.tokens, self.hidden) or lengths.shape != (len(prefix),):
            raise ValueError('Expected padded prefix and lengths')
        values = ((prefix - self.mean) / self.scale).flatten(1) - self.center
        latent = (values @ self.basis.T) / self.latent_scale
        return torch.cat([latent, (2 * lengths[:, None] / self.tokens - 1)], -1)

    def unpack(self, state):
        values = (state[:, :-1] * self.latent_scale) @ self.basis + self.center
        prefix = values.reshape(-1, self.tokens, self.hidden) * self.scale + self.mean
        lengths = ((state[:, -1] + 1) * self.tokens / 2).round().long().clamp(1, self.tokens)
        return prefix, lengths

    @torch.no_grad()
    def sample(self, features, seed=42, steps=64):
        if steps < 1:
            raise ValueError('Positive integration steps required')
        generator = torch.Generator(device=features.device).manual_seed(seed)
        state = torch.randn(len(features), self.dimension + 1,
                            generator=generator, device=features.device, dtype=features.dtype)
        for i in range(steps):
            time = features.new_full((len(features),), i / steps)
            state = state + self.velocity(state, features, time) / steps
        if not torch.isfinite(state).all():
            raise RuntimeError('Nonfinite generated conditioning')
        return self.unpack(state)


def flow_loss(prediction, velocity):
    if prediction.shape != velocity.shape or prediction.ndim != 2:
        raise ValueError('Expected aligned flow velocities')
    error = (prediction - velocity).square()
    return error[:, :-1].mean() + error[:, -1].mean()
