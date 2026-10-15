"""Causal continuous prefix prior; never predicts or consumes variable text IDs."""
import math

import torch
from torch import nn


class AutoregressivePrior(nn.Module):
    def __init__(self, mean, scale, basis, latent_scale, hidden=256, width=512, maximum=32):
        super().__init__()
        if basis.ndim != 2 or basis.shape[1] != len(mean) or scale.shape != mean.shape:
            raise ValueError('Invalid continuous token subspace')
        if latent_scale.shape != (len(basis),) or not bool((latent_scale > 0).all() and (scale > 0).all()):
            raise ValueError('Invalid normalization')
        if not all(torch.isfinite(t).all() for t in [mean, scale, basis, latent_scale]):
            raise ValueError('Finite normalization required')
        for key, value in [('mean', mean), ('scale', scale), ('basis', basis), ('latent_scale', latent_scale)]:
            self.register_buffer(key, value.clone())
        self.dimension, self.maximum = len(basis), maximum
        self.features = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 128))
        self.previous = nn.Linear(self.dimension, 128)
        self.position = nn.Embedding(maximum + 1, 128)
        self.gru = nn.GRU(128, hidden, num_layers=2, batch_first=True)
        self.stop = nn.Linear(hidden, 1)
        self.flow_state = nn.Linear(self.dimension, width)
        self.flow_context = nn.Linear(hidden, width)
        self.flow_time = nn.Sequential(nn.Linear(16, width), nn.SiLU(), nn.Linear(width, width))
        self.flow_body = nn.Sequential(nn.SiLU(), nn.Linear(width, width), nn.SiLU(),
                                       nn.Linear(width, width), nn.SiLU(), nn.Linear(width, self.dimension))

    def encode(self, prefix):
        return (((prefix - self.mean) / self.scale) @ self.basis.T) / self.latent_scale

    def decode(self, latent):
        return ((latent * self.latent_scale) @ self.basis) * self.scale + self.mean

    def contexts(self, features, shifted_previous):
        if features.ndim != 2 or features.shape[1] != 768 or shifted_previous.ndim != 3:
            raise ValueError('Expected numeric semantic/image features and shifted continuous vectors')
        if shifted_previous.shape[0] != len(features) or shifted_previous.shape[2] != self.dimension:
            raise ValueError('Previous-vector dimensions differ')
        positions = torch.arange(shifted_previous.shape[1], device=features.device)
        inputs = (self.previous(shifted_previous) + self.features(features)[:, None]
                  + self.position(positions)[None])
        return self.gru(inputs)[0]

    def velocity(self, state, context, time):
        frequencies = 2 ** torch.arange(8, device=time.device, dtype=time.dtype)
        phases = time[:, None] * frequencies * math.pi
        temporal = self.flow_time(torch.cat([phases.sin(), phases.cos()], -1))
        learned = self.flow_body(self.flow_state(state) + self.flow_context(context) + temporal)
        coefficient = (2 * time - 1) / (time.square() + (1 - time).square())
        return learned + coefficient[:, None] * state

    @torch.no_grad()
    def sample(self, features, seed=42, steps=64):
        if features.shape != (1, 768) or not torch.isfinite(features).all() or steps < 1:
            raise ValueError('Finite batch-one image/semantic vectors and positive steps required')
        generator = torch.Generator(device=features.device).manual_seed(seed)
        previous = features.new_zeros(1, self.dimension)
        recurrent, values, probabilities = None, [], []
        semantic = self.features(features)
        for i in range(self.maximum + 1):
            context_input = (self.previous(previous) + semantic + self.position.weight[i])[None]
            # context_input [1,B=1,128] is batch-first for this batch-one path.
            context, recurrent = self.gru(context_input, recurrent)
            stop = float(self.stop(context[:, 0]).sigmoid()[0, 0])
            probabilities.append(stop)
            if (i > 0 and stop >= 0.5) or i == self.maximum:
                break
            state = torch.randn(1, self.dimension, generator=generator, device=features.device)
            for step in range(steps):
                time = features.new_full((1,), step / steps)
                state = state + self.velocity(state, context[:, 0], time) / steps
            if not torch.isfinite(state).all():
                raise RuntimeError('Nonfinite continuous prefix sample')
            values.append(state);previous = state
        latent = torch.stack(values, 1)
        return self.decode(latent), probabilities


def causal_loss(model, features, target, lengths):
    clean = model.encode(target)
    shifted = torch.cat([torch.zeros_like(clean[:, :1]), clean], 1)
    noise = torch.randn_like(shifted) * 0.02
    noise[:, 0] = 0
    context = model.contexts(features, shifted + noise)
    positions = torch.arange(model.maximum + 1, device=features.device)[None]
    valid = positions[:, :-1] < lengths[:, None]
    gold = clean[valid]
    gaussian = torch.randn_like(gold)
    time = torch.rand(len(gold), device=features.device)
    interpolated = (1 - time[:, None]) * gaussian + time[:, None] * gold
    velocity = model.velocity(interpolated, context[:, :-1][valid], time)
    flow = (velocity - (gold - gaussian)).square().mean()
    stop_mask = positions <= lengths[:, None]
    stop_targets = (positions == lengths[:, None]).float()
    eos = nn.functional.binary_cross_entropy_with_logits(model.stop(context).squeeze(-1)[stop_mask],
                                                         stop_targets[stop_mask])
    return flow + 0.2 * eos, {'velocity_loss': flow, 'stop_bce': eos}
