"""Text-free numeric continuation inference for the bounded T0/T1 bridge."""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from latent_diaries.inversion import digest, load_backbone, render
from latent_diaries.temporal_reader import TemporalReader


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["run-dir", "inputs", "output-dir"]:
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    if not Path("/.dockerenv").exists() or args.output_dir.exists():
        raise ValueError("Managed worker and fresh output directory required")
    data = json.loads(args.inputs.read_text())
    if set(data) != {"rows"} or not data["rows"]:
        raise ValueError("Strict numeric-only input required")
    for row in data["rows"]:
        if set(row) != {"id", "image", "audio", "present"} or not isinstance(row["present"], bool):
            raise ValueError(
                "Numeric media vectors and history-presence only; no text or teacher targets"
            )
    if len({r["id"] for r in data["rows"]}) != len(data["rows"]):
        raise ValueError("Duplicate output identity")
    info = json.loads((args.run_dir / "config.json").read_text())
    if (
        digest(args.run_dir / "reader.safetensors") != info["reader_sha256"]
        or digest(Path("uv.lock")) != info["lock_sha256"]
        or digest(Path("src/latent_diaries/temporal_reader.py")) != info["reader_source_sha256"]
    ):
        raise ValueError("Checkpoint, architecture or lock changed")
    weights = load_file(str(args.run_dir / "reader.safetensors"))
    reader = TemporalReader(
        info["maximum_tokens"],
        weights["mean"],
        weights["scale"],
        feedback=info["feedback"],
        width=info["width"],
    )
    reader.load_state_dict(weights)
    wrapper, builder, speaker = load_backbone(info)
    reader.to(speaker.device).eval()
    args.output_dir.mkdir(mode=0o700)
    records = []
    with torch.no_grad():
        for i, row in enumerate(data["rows"]):
            image = torch.tensor([row["image"]], device=speaker.device)
            audio = torch.tensor([row["audio"]], device=speaker.device)
            present = torch.tensor([row["present"]], device=speaker.device)
            prefix, mask, lengths, continuous = reader(image, audio, present)
            length = int(lengths[0])
            if not bool(mask[0, :length].all()):
                raise ValueError("Contiguous prefix mask required")
            output = render(
                builder,
                wrapper,
                prefix[:, :length],
                speaker,
                args.output_dir / f"{i:03}.wav",
                240,
                42,
                codec0_greedy=True,
            )
            records.append(
                {
                    "id": row["id"],
                    "prefix_length": length,
                    "continuous_length": float(continuous[0]),
                    **output,
                }
            )
            # Preserve completed/capped outputs; never choose a different seed.
            result = {
                "scope": "Bounded teacher-history continuation bridge; not autonomous two-fragment rollout",
                "records": records,
                "reader_sha256": info["reader_sha256"],
                "inputs_sha256": digest(args.inputs),
                "source_sha256": digest(Path(__file__)),
                "text_used": False,
                "teacher_targets_read": False,
                "backbones_frozen": True,
                "codec0_greedy": True,
                "residuals_sampled": True,
                "seed": 42,
            }
            (args.output_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n")
            print(
                json.dumps({"complete": len(records), "id": row["id"], "frames": output["frames"]}),
                flush=True,
            )


if __name__ == "__main__":
    main()
