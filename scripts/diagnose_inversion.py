"""Measure frozen residual-codebook behavior after codec-0-only inversion."""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from latent_diaries.inversion import condition, load_backbone


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    metadata = json.loads((args.run_dir / "config.json").read_text())
    wrapper, builder, speaker = load_backbone(metadata)
    prefix = load_file(str(args.run_dir / "prefix.safetensors"))["soft_tokens"].to(speaker.device)
    gold = torch.load(args.run_dir / "gold_codes.pt", map_location=speaker.device, weights_only=True)
    with torch.no_grad():
        packed = builder.build_teacher_forced(condition(prefix, speaker), gold)
        output = wrapper.model.talker(**packed, output_hidden_states=True)
        length = builder.build_prefill(condition(prefix, speaker))["inputs_embeds"].shape[1]
        codec0_logits = output.logits[:, length - 1 : length - 1 + gold.shape[1]]
        hidden = output.hidden_states[0][-1][:, length - 1 : length - 1 + gold.shape[1]]
        residual_logits, _ = wrapper.model.talker.forward_sub_talker_finetune(gold[0], hidden[0])
        # This head predicts each residual codebook at the same position.
        # Do not reuse the upstream generic causal loss (which shifts labels).
        loss = torch.nn.functional.cross_entropy(
            residual_logits.reshape(-1, residual_logits.shape[-1]), gold[0, :, 1:].reshape(-1)
        )
        result = {
            "teacher_forced_codec0_loss": float(output.loss),
            "teacher_forced_codec0_accuracy": float(
                (codec0_logits.argmax(-1) == gold[:, :, 0]).float().mean()
            ),
            "same_position_residual_loss": float(loss),
            "same_position_residual_accuracy": float(
                (residual_logits.argmax(-1) == gold[0, :, 1:]).float().mean()
            ),
            "frames": gold.shape[1],
            "residual_codebooks": residual_logits.shape[1],
            "alignment": "Frame t hidden at prefix_length-1+t; predicts codebooks 1..15 with gold previous codebooks",
            "scope": "Teacher-forced diagnostic only; no speech quality assertion",
        }
    (args.run_dir / "diagnostic.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
