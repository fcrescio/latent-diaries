"""Read-only post-test prefix diagnostics; teacher vectors never condition inference."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from latent_diaries.prefix_regression import predict_prefix
from latent_diaries.shared_mapper import validate_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not Path("/.dockerenv").exists():
        raise RuntimeError("Use the managed worker")
    rows = json.loads(args.dataset.read_text())["rows"]
    train, _ = validate_split(rows)
    info = json.loads((args.run_dir / "config.json").read_text())
    path = args.run_dir / "mapper.safetensors"
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    if checksum != info["mapper_sha256"]:
        raise ValueError("Mapper checksum differs")
    weights = load_file(str(path))
    results = []
    for row in rows:
        features = torch.tensor(row["features"], dtype=torch.float32)
        predicted, length = predict_prefix(features, weights["prefix_coef"], weights["length_coef"])
        gold = load_file(str(Path(row["cache_dir"]) / "prefix.safetensors"))["soft_tokens"]
        aligned = min(length, gold.shape[1])
        error = (predicted[0, :aligned] - gold[0, :aligned]).square().mean(-1)
        initial = min(5, aligned)
        result = {
            "id": row["id"],
            "split": row["split"],
            "predicted_prefix_length": length,
            "teacher_prefix_length": gold.shape[1],
            "length_matches": length == gold.shape[1],
            "prefix_mse": float(error.mean()),
            "first_five_positions_mse": float(error[:initial].mean()),
            "remaining_positions_mse": float(error[initial:].mean()) if initial < aligned else None,
            "per_position_mse": error.tolist(),
        }
        results.append(result)
    features = torch.tensor([[1.0, *r["features"]] for r in train])
    summary = {
        "scope": "Post-test teacher-prefix diagnostic, not a speech-quality score or word alignment",
        "coefficients_updated": False,
        "all_lengths_match": all(r["length_matches"] for r in results),
        "train_design_columns": features.shape[1],
        "train_design_rank": int(torch.linalg.matrix_rank(features)),
        "results": results,
    }
    assert hashlib.sha256(path.read_bytes()).hexdigest() == checksum
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}), flush=True)
    print(
        json.dumps([{k: v for k, v in r.items() if k != "per_position_mse"} for r in results]),
        flush=True,
    )


if __name__ == "__main__":
    main()
