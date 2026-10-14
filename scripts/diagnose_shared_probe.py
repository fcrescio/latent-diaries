"""Post-training teacher-forced diagnostics; never used to select/update weights."""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from latent_diaries.inversion import digest, distribution_kl, load_backbone, trajectory_logits
from latent_diaries.shared_mapper import SharedMapper, validate_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads(args.dataset.read_text())["rows"]
    validate_split(rows)
    info = json.loads((args.run_dir / "config.json").read_text())
    checkpoint = args.run_dir / "mapper.safetensors"
    if digest(checkpoint) != info["mapper_sha256"]:
        raise ValueError("Mapper checksum differs")
    weights = load_file(str(checkpoint))
    mapper = SharedMapper(weights["base"], info["input_dim"], info["width"])
    mapper.load_state_dict(weights)
    wrapper, builder, speaker = load_backbone(info)
    mapper = mapper.to(speaker.device).eval().requires_grad_(False)
    results = []
    with torch.no_grad():
        for row in rows:
            path = Path(row["cache_dir"]) / "teacher.safetensors"
            if digest(path) != row["cache_sha256"]:
                raise ValueError("Teacher checksum differs")
            cache = {k: v.to(speaker.device) for k, v in load_file(str(path)).items()}
            features = torch.tensor([row["features"]], device=speaker.device, dtype=torch.float32)
            for name, value in [("conditioned", features), ("null", torch.zeros_like(features))]:
                first, residual = trajectory_logits(
                    wrapper, builder, mapper(value), speaker, cache["audio_codes"]
                )
                results.append(
                    {
                        "id": row["id"],
                        "split": row["split"],
                        "condition": name,
                        "codec0_kl": float(distribution_kl(first, cache["codec0_logits"])),
                        "residual_kl": float(distribution_kl(residual, cache["residual_logits"])),
                    }
                )
    assert digest(checkpoint) == info["mapper_sha256"]
    args.output.write_text(
        json.dumps(
            {
                "scope": "Post-training teacher-forced diagnostic, not autoregressive speech quality",
                "weights_updated": False,
                "results": results,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
