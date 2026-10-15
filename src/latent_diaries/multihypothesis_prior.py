"""Context-dependent continuous modes; no text IDs, vocabulary or sentence lookup."""
import torch
from torch import nn

from .autoregressive_prior import AutoregressivePrior


class MultihypothesisPrior(AutoregressivePrior):
    def __init__(self, mean, scale, basis, latent_scale, modes=8, hidden=256, maximum=32):
        super().__init__(mean, scale, basis, latent_scale, hidden=hidden, maximum=maximum)
        for name in ['flow_state', 'flow_context', 'flow_time', 'flow_body']:
            delattr(self, name)
        self.modes = modes
        self.candidates = nn.Sequential(nn.Linear(hidden, 512), nn.SiLU(),
                                        nn.Linear(512, modes * self.dimension))
        self.routing = nn.Linear(hidden, modes)

    def predictions(self, context):
        means = self.candidates(context).reshape(*context.shape[:-1], self.modes, self.dimension)
        return means, self.routing(context)

    @torch.no_grad()
    def sample(self, features, seed=42, steps=64):
        # Greedy continuous mode is deterministic; seed is retained for caller parity.
        if features.shape != (1, 768) or not torch.isfinite(features).all():
            raise ValueError('Finite batch-one numeric features required')
        previous = features.new_zeros(1, self.dimension)
        recurrent, values, probabilities = None, [], []
        semantic = self.features(features)
        for i in range(self.maximum + 1):
            context_input = (self.previous(previous) + semantic + self.position.weight[i])[:, None]
            context, recurrent = self.gru(context_input, recurrent)
            context = context[:, 0]
            probability = float(self.stop(context).sigmoid()[0, 0]);probabilities.append(probability)
            if (i > 0 and probability >= 0.5) or i == self.maximum:
                break
            means, routing = self.predictions(context)
            previous = means[0, int(routing[0].argmax())][None]
            values.append(previous)
        return self.decode(torch.stack(values, 1)), probabilities


def multihypothesis_loss(model, features, target, lengths):
    clean = model.encode(target)
    previous = torch.cat([torch.zeros_like(clean[:, :1]), clean], 1)
    perturbation = torch.randn_like(previous) * 0.02;perturbation[:, 0] = 0
    context = model.contexts(features, previous + perturbation)
    positions = torch.arange(model.maximum + 1, device=features.device)[None]
    valid = positions[:, :-1] < lengths[:, None]
    means, routing = model.predictions(context[:, :-1][valid])
    # Orthonormal basis makes this equal to normalized-prefix reconstruction.
    error = ((means - clean[valid][:, None]) * model.latent_scale).square().sum(-1) / model.mean.numel()
    best_error, winner = error.min(-1)
    reconstruction = best_error.mean()
    mode_loss = nn.functional.cross_entropy(routing, winner.detach())
    stop_mask = positions <= lengths[:, None]
    stop_targets = (positions == lengths[:, None]).float()
    eos = nn.functional.binary_cross_entropy_with_logits(model.stop(context).squeeze(-1)[stop_mask],
                                                         stop_targets[stop_mask])
    return reconstruction + 0.1 * mode_loss + 0.2 * eos, {
        'best_normalized_prefix_mse': reconstruction, 'routing_ce': mode_loss, 'stop_bce': eos}
