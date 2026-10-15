"""Context-dependent continuous modes; no text IDs, vocabulary or sentence lookup."""
import torch
from torch import nn

from .autoregressive_prior import AutoregressivePrior


def continuous_log_ratio(means, routing, null_means, null_routing, latent_scale,
                         normalization_dimension, temperature=0.01):
    """Density ratio on actual vectors; internal mode indices need not align."""
    conditional_error = ((means[:, None] - means[None]) * latent_scale).square().sum(-1) / normalization_dimension
    null_error = ((means[:, None] - null_means[None]) * latent_scale).square().sum(-1) / normalization_dimension
    conditional = torch.logsumexp(routing.log_softmax(-1)[None] - conditional_error / temperature, -1)
    unconditional = torch.logsumexp(null_routing.log_softmax(-1)[None] - null_error / temperature, -1)
    return (conditional - unconditional).clamp(-2, 2)


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
    def sample(self, features, seed=42, steps=64, sample_modes=False, temperature=0.5,
               guidance_scale=0.0, guidance_steps=12):
        if features.shape != (1, 768) or not torch.isfinite(features).all():
            raise ValueError('Finite batch-one numeric features required')
        if not 0 <= guidance_scale <= 2 or guidance_steps < 0:
            raise ValueError('Guidance scale must be bounded and window nonnegative')
        previous = features.new_zeros(1, self.dimension)
        generator = torch.Generator(device=features.device).manual_seed(seed)
        recurrent, values, probabilities = None, [], []
        semantic = self.features(features)
        null_semantic = self.features(torch.zeros_like(features)) if guidance_scale else None
        null_recurrent = None
        for i in range(self.maximum + 1):
            context_input = (self.previous(previous) + semantic + self.position.weight[i])[:, None]
            context, recurrent = self.gru(context_input, recurrent)
            context = context[:, 0]
            probability = float(self.stop(context).sigmoid()[0, 0]);probabilities.append(probability)
            if (i > 0 and probability >= 0.5) or i == self.maximum:
                break
            means, routing = self.predictions(context)
            if guidance_scale and i < guidance_steps:
                null_input = (self.previous(previous) + null_semantic + self.position.weight[i])[:, None]
                null_context, null_recurrent = self.gru(null_input, null_recurrent)
                null_means, null_routing = self.predictions(null_context[:, 0])
                ratio = continuous_log_ratio(means[0], routing[0], null_means[0], null_routing[0],
                                             self.latent_scale, self.mean.numel())
                logits, indices = routing[0].topk(min(4, self.modes))
                logits = logits + guidance_scale * ratio[indices]
                selected = (int(torch.multinomial((logits / temperature).softmax(-1), 1, generator=generator))
                            if sample_modes else int(logits.argmax()))
                index = int(indices[selected])
            elif sample_modes:
                logits, indices = routing[0].topk(min(4, self.modes))
                selected = int(torch.multinomial((logits / temperature).softmax(-1), 1, generator=generator))
                index = int(indices[selected])
            else:
                index = int(routing[0].argmax())
            previous = means[0, index][None]
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
