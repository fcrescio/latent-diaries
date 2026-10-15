"""Teacher-only geometric diagnostic, never a learned speech or inference path."""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from latent_diaries.shared_mapper import validate_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not Path("/.dockerenv").exists():
        raise RuntimeError("Use the managed worker")
    rows = json.loads(args.dataset.read_text())["rows"]
    validate_split(rows)
    features = torch.tensor([[1.0, *r["features"]] for r in rows], dtype=torch.float64)
    train = torch.tensor([r["split"] == "train" for r in rows])
    prefixes = [load_file(str(Path(r["cache_dir"]) / "prefix.safetensors"))["soft_tokens"] for r in rows]
    targets = {
        "mean_pool": torch.cat([p.mean(1) for p in prefixes]).double(),
        "resampled_32_tokens": torch.cat(
            [
                torch.nn.functional.interpolate(
                    p.transpose(1, 2), size=32, mode="linear", align_corners=False
                )
                .transpose(1, 2)
                .reshape(1, -1)
                for p in prefixes
            ]
        ).double(),
    }
    results = {}
    for name, target in targets.items():
        # Fit only six train rows; exposed development rows diagnose the residual.
        weights = torch.linalg.lstsq(features[train], target[train]).solution
        predicted = features @ weights
        mse = (predicted - target).square().mean(-1)
        baseline = (target - target[train].mean(0)).square().mean(-1)
        results[name] = {
            "train_mse": float(mse[train].mean()),
            "development_mse": float(mse[~train].mean()),
            "development_constant_baseline_mse": float(baseline[~train].mean()),
        }
    args.output.write_text(
        json.dumps(
            {
                "scope": "Teacher-prefix geometry; no speech-quality or model-success claim",
                "teacher_prefix_lengths": {
                    r["id"]: p.shape[1] for r, p in zip(rows, prefixes, strict=True)
                },
                "train_rows_used_in_fit": [r["id"] for r in rows if r["split"] == "train"],
                "results": results,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
