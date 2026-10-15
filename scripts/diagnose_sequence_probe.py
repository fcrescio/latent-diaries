"""Read-only post-training sequence-prefix geometry, not perceptual scoring."""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from sequence_probe import make_reader

from latent_diaries.inversion import digest
from latent_diaries.shared_mapper import validate_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not Path("/.dockerenv").exists():
        raise RuntimeError("Use managed worker")
    rows = json.loads(args.dataset.read_text())["rows"]
    validate_split(rows)
    info = json.loads((args.run_dir / "config.json").read_text())
    path = args.run_dir / "reader.safetensors"
    if digest(path) != info["reader_sha256"]:
        raise ValueError("Reader checksum differs")
    reader = make_reader(info, load_file(str(path))).eval().requires_grad_(False)
    results = []
    with torch.no_grad():
        for row in rows:
            prefix, _, lengths = reader(torch.tensor([row["features"]], dtype=torch.float32))
            gold = load_file(str(Path(row["cache_dir"]) / "prefix.safetensors"))["soft_tokens"]
            aligned = min(int(lengths[0]), gold.shape[1])
            error = (prefix[0, :aligned] - gold[0, :aligned]).square().mean(-1)
            results.append(
                {
                    "id": row["id"],
                    "split": row["split"],
                    "predicted_prefix_length": int(lengths[0]),
                    "teacher_prefix_length": gold.shape[1],
                    "length_matches": int(lengths[0]) == gold.shape[1],
                    "prefix_mse": float(error.mean()),
                    "first_five_positions_mse": float(error[:5].mean()),
                    "remaining_positions_mse": float(error[5:].mean()),
                }
            )
    assert digest(path) == info["reader_sha256"]
    summary = {}
    for split in ["train", "heldout"]:
        selected = [r for r in results if r["split"] == split]
        summary[split] = {
            k: sum(r[k] for r in selected) / len(selected)
            for k in ["prefix_mse", "first_five_positions_mse", "remaining_positions_mse"]
        }
    args.output.write_text(
        json.dumps(
            {
                "scope": "Frozen post-training teacher-prefix diagnostic, not audio quality or word alignment",
                "weights_updated": False,
                "summary": summary,
                "results": results,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
