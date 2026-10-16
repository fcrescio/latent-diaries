"""Explicit low-rank updates on final Talker q/v projections; no vendor copies."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    def __init__(self, base, rank):
        super().__init__()
        if not isinstance(base, nn.Linear) or not 1 <= rank <= min(base.in_features, base.out_features):
            raise ValueError('Compatible linear layer and positive bounded rank required')
        self.base = base.requires_grad_(False)
        self.in_features, self.out_features = base.in_features, base.out_features
        self.lora_a = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.lora_b = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        nn.init.uniform_(self.lora_a, -1 / math.sqrt(base.in_features), 1 / math.sqrt(base.in_features))
        self.scale = 1.0  # alpha equals rank; initial B=0 preserves the original function.

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, values):
        return self.base(values) + F.linear(F.linear(values, self.lora_a), self.lora_b) * self.scale


def inject_talker_lora(talker, *, rank=8, layers=4):
    if any(p.requires_grad for p in talker.parameters()):
        raise ValueError('Freeze the original Talker before injecting LoRA')
    blocks = talker.model.layers
    if not 1 <= layers <= len(blocks):
        raise ValueError('Requested final attention layers unavailable')
    selected = []
    for i in range(len(blocks)-layers, len(blocks)):
        for projection in ['q_proj', 'v_proj']:
            base = getattr(blocks[i].self_attn, projection)
            if not isinstance(base, nn.Linear) or not 1 <= rank <= min(base.in_features, base.out_features):
                raise ValueError('Pinned Talker projection layout differs')
            selected.append((i, projection, base))
    manifest = []
    for i, name, base in selected:
        setattr(blocks[i].self_attn, name, LoRALinear(base, rank))
        manifest.append({'module': f'model.layers.{i}.self_attn.{name}',
                         'base_shape': list(base.weight.shape), 'rank': rank, 'scale': 1.0})
    return manifest


def lora_parameters(talker):
    return [p for name, p in talker.named_parameters() if name.endswith(('.lora_a', '.lora_b'))]


def lora_state(talker):
    return {name: p.detach().cpu().contiguous() for name, p in talker.named_parameters()
            if name.endswith(('.lora_a', '.lora_b'))}


def load_lora_state(talker, weights):
    expected = {name: p for name, p in talker.named_parameters() if name.endswith(('.lora_a', '.lora_b'))}
    if not expected or set(weights) != set(expected):
        raise ValueError('Exact LoRA-only tensor keys required')
    if any(weights[k].shape != p.shape or not torch.isfinite(weights[k]).all()
           for k, p in expected.items()):
        raise ValueError('LoRA tensor shape/value differs')
    with torch.no_grad():
        for key, param in expected.items():
            param.copy_(weights[key].to(device=param.device, dtype=param.dtype))
