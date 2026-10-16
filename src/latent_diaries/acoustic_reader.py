"""Acoustic supervision through frozen Qwen; text targets are training-only."""
from torch.nn import functional as F

from .inversion import aligned_residual_logits, condition


def acoustic_objective(wrapper, builder, prefix, speaker, gold, target, scale, *, prefix_weight=0.1):
    if prefix.shape != target.shape or gold.shape[0] != 1:
        raise ValueError('Expected batch-one aligned acoustic/prefix targets')
    base_loss, metrics = codec_objective(wrapper, builder, prefix, speaker, gold)
    reconstruction = ((prefix - target.detach()) / scale.detach()).square().mean()
    if prefix_weight < 0:
        raise ValueError("Nonnegative prefix weight required")
    loss = base_loss + prefix_weight * reconstruction
    return loss, {**metrics, 'normalized_prefix_mse': reconstruction}


def codec_objective(wrapper, builder, prefix, speaker, gold):
    """Actual speech loss with no sentence-vector target or text-length coupling."""
    if prefix.shape[0] != 1 or gold.shape[0] != 1:
        raise ValueError('This aligned residual objective requires batch one')
    visual = condition(prefix, speaker)
    batch = builder.build_teacher_forced(visual, gold)
    output = wrapper.model.talker(**batch, output_hidden_states=True)
    prefill_length = builder.build_prefill(visual)['inputs_embeds'].shape[1]
    hidden = output.hidden_states[0][-1][0, prefill_length - 1:prefill_length - 1 + gold.shape[1]]
    residual_logits = aligned_residual_logits(wrapper.model.talker, hidden, gold[0])
    residual = F.cross_entropy(residual_logits.reshape(-1, residual_logits.shape[-1]),
                               gold[0, :, 1:].reshape(-1))
    return output.loss + 0.1 * residual, {'codec0_ce': output.loss, 'residual_ce': residual}
