"""Gold-code teacher forcing and frozen-Qwen gradient checks, offline only."""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from latent_diaries.acoustic_reader import codec_objective
from latent_diaries.inversion import digest, load_backbone
from latent_diaries.temporal_reader import TemporalReader


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--targets", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, action="append", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if not Path("/.dockerenv").exists() or args.output.exists():
        raise ValueError("Managed worker and fresh diagnostic output required")
    data = json.loads(args.targets.read_text())
    records = []
    wrapper, builder, speaker = load_backbone(data)
    for run in args.run_dir:
        info = json.loads((run / "config.json").read_text())
        if (
            info["targets_sha256"] != digest(args.targets)
            or digest(run / "reader.safetensors") != info["reader_sha256"]
        ):
            raise ValueError("Training targets or checkpoint changed")
        weights = load_file(str(run / "reader.safetensors"))
        reader = TemporalReader(
            info["maximum_tokens"],
            weights["mean"],
            weights["scale"],
            feedback=info["feedback"],
            width=info["width"],
        ).to(speaker.device)
        reader.load_state_dict(weights)
        for split in ["train", "validation"]:
            row = min((r for r in data["records"] if r["split"] == split), key=lambda r: r["id"])
            for key in ["prefix", "codes"]:
                if digest(Path(row[key])) != row[key + "_sha256"]:
                    raise ValueError("Diagnostic gold target changed")
            gold = load_file(row["codes"])["audio_codes"].to(speaker.device)
            with torch.no_grad():
                oracle = load_file(row["prefix"])["soft_tokens"].to(speaker.device)
                _, teacher_metrics = codec_objective(wrapper, builder, oracle, speaker, gold)
            image = torch.tensor([row["image"]], device=speaker.device)
            audio = torch.tensor([row["audio"]], device=speaker.device)
            prefix, _, lengths, _ = reader(
                image, audio, torch.ones(1, dtype=torch.bool, device=speaker.device)
            )
            loss, metrics = codec_objective(
                wrapper, builder, prefix[:, : int(lengths[0])], speaker, gold
            )
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(reader.parameters(), 1.0)
            if not torch.isfinite(norm) or norm <= 0:
                raise RuntimeError("Missing/nonfinite gradients through frozen Talker")
            if any(p.requires_grad or p.grad is not None for p in wrapper.model.parameters()):
                raise RuntimeError("Backbone freezing violated")
            records.append(
                {
                    "feedback": info["feedback"],
                    "split": split,
                    "id": row["id"],
                    "student": {k: float(v.detach()) for k, v in metrics.items()},
                    "oracle": {k: float(v.detach()) for k, v in teacher_metrics.items()},
                    "reader_gradient_norm": float(norm),
                    "backbone_gradients_absent": True,
                }
            )
            reader.zero_grad(set_to_none=True)
    result = {
        "scope": "Auxiliary teacher-forced acoustic and gradient diagnostic, not artistic/intelligibility scores",
        "records": records,
        "optimizer_steps": 0,
        "gold_previous_audio_codes": True,
        "source_sha256": digest(Path(__file__)),
        "targets_sha256": digest(args.targets),
        "peak_vram_bytes": torch.cuda.max_memory_allocated(),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"gradient_controls": len(records), "backbones_frozen": True}))


if __name__ == "__main__":
    main()
