"""Compare teacher/student policies on gold versus student-generated histories."""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from latent_diaries.inversion import (
    aligned_residual_logits,
    condition,
    distribution_kl,
    frame_logits,
    load_backbone,
)


def policies(wrapper, builder, prefix, speaker, codes):
    output = wrapper.model.talker(
        **builder.build_teacher_forced(condition(prefix, speaker), codes), output_hidden_states=True
    )
    length = builder.build_prefill(condition(prefix, speaker))["inputs_embeds"].shape[1]
    hidden = output.hidden_states[0][-1][0, length - 1 : length - 1 + codes.shape[1]]
    return frame_logits(output.logits, length, codes.shape[1]), aligned_residual_logits(
        wrapper.model.talker, hidden, codes[0]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--teacher-cache-dir", type=Path, required=True)
    args = parser.parse_args()
    metadata = json.loads((args.run_dir / "config.json").read_text())
    wrapper, builder, speaker = load_backbone(metadata)
    prefix = load_file(str(args.run_dir / "prefix.safetensors"))["soft_tokens"].to(speaker.device)
    teacher = load_file(str(args.teacher_cache_dir / "prefix.safetensors"))["soft_tokens"].to(
        speaker.device
    )
    gold = load_file(str(args.teacher_cache_dir / "teacher.safetensors"))["audio_codes"].to(
        speaker.device
    )
    generated = builder.generate_codes(
        condition(prefix, speaker), max_frames=gold.shape[1] + 24, seed=metadata["seed"]
    )[0][None]
    results = {}
    with torch.no_grad():
        for name, codes in [("gold_history", gold), ("student_history", generated)]:
            t0, tr = policies(wrapper, builder, teacher, speaker, codes)
            s0, sr = policies(wrapper, builder, prefix, speaker, codes)
            per_frame = torch.nn.functional.kl_div(
                torch.log_softmax(s0.float(), -1),
                torch.log_softmax(t0.float(), -1),
                log_target=True,
                reduction="none",
            ).sum(-1)[0]
            results[name] = {
                "frames": codes.shape[1],
                "codec0_kl": float(distribution_kl(s0, t0)),
                "residual_kl": float(distribution_kl(sr, tr)),
                "codec0_kl_per_frame": per_frame.tolist(),
            }
    (args.run_dir / "policy-gap.json").write_text(json.dumps(results, indent=2) + "\n")
    print(
        json.dumps(
            {k: {x: y for x, y in v.items() if x != "codec0_kl_per_frame"} for k, v in results.items()}
        )
    )


if __name__ == "__main__":
    main()
