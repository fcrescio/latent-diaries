"""Acoustic supervision through frozen Qwen; text targets are training-only."""
from torch.nn import functional as F

from .inversion import aligned_residual_logits, condition


def acoustic_objective(wrapper, builder, prefix, speaker, gold, target, scale):
    if prefix.shape != target.shape or gold.shape[0] != 1:
        raise ValueError('Expected batch-one aligned acoustic/prefix targets')
    visual = condition(prefix, speaker)
    batch = builder.build_teacher_forced(visual, gold)
    output = wrapper.model.talker(**batch, output_hidden_states=True)
    prefill_length = builder.build_prefill(visual)['inputs_embeds'].shape[1]
    hidden = output.hidden_states[0][-1][0, prefill_length - 1:prefill_length - 1 + gold.shape[1]]
    residual_logits = aligned_residual_logits(wrapper.model.talker, hidden, gold[0])
    residual = F.cross_entropy(residual_logits.reshape(-1, residual_logits.shape[-1]),
                               gold[0, :, 1:].reshape(-1))
    reconstruction = ((prefix - target.detach()) / scale.detach()).square().mean()
    loss = output.loss + 0.1 * residual + 0.1 * reconstruction
    return loss, {'codec0_ce': output.loss, 'residual_ce': residual,
                  'normalized_prefix_mse': reconstruction}
