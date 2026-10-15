"""Direct training-only teacher-prefix supervision; numeric-only evaluation.

Right-padded continuous teacher vectors and prefix length are fit jointly from
six train combinations. No per-example lookup, text IDs or tokenizer at eval.
The exposed development combinations are not a fresh test set.
"""

import argparse
import json
import os
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from latent_diaries.inversion import digest, load_backbone, render
from latent_diaries.prefix_regression import predict_prefix
from latent_diaries.shared_mapper import validate_split


def fit(args):
    rows, dev = validate_split(json.loads(args.dataset.read_text())["rows"])
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        raise ValueError("Use a fresh run directory")
    prefixes = []
    infos = []
    for row in rows:
        path = Path(row["cache_dir"])
        infos.append(json.loads((path / "config.json").read_text()))
        prefixes.append(load_file(str(path / "prefix.safetensors"))["soft_tokens"])
    for info in infos:
        for key in ["model_id", "model_revision", "lock_sha256", "role_ids"]:
            if info[key] != infos[0][key]:
                raise ValueError("Teacher provenance differs")
    lengths = torch.tensor([p.shape[1] for p in prefixes], dtype=torch.float64)
    maximum = int(lengths.max())
    target = torch.zeros(len(rows), maximum, prefixes[0].shape[-1], dtype=torch.float64)
    for i, prefix in enumerate(prefixes):
        target[i, : prefix.shape[1]] = prefix[0]
    features = torch.tensor([[1.0, *r["features"]] for r in rows], dtype=torch.float64)
    coef = torch.linalg.lstsq(features, target.flatten(1)).solution
    length_coef = torch.linalg.lstsq(features, lengths[:, None]).solution[:, 0]
    args.run_dir.mkdir(parents=True, exist_ok=True)
    weights = {
        "prefix_coef": coef.reshape(features.shape[1], maximum, -1).float().contiguous(),
        "length_coef": length_coef.float().contiguous(),
    }
    save_file(weights, str(args.run_dir / "mapper.safetensors"))
    info = {k: infos[0][k] for k in ["model_id", "model_revision", "lock_sha256"]}
    info.update(
        source_revision=os.environ.get("LD_CODE_REVISION"),
        source_sha256=digest(Path(__file__)),
        reader_source_sha256=digest(Path("src/latent_diaries/prefix_regression.py")),
        dataset_sha256=digest(args.dataset),
        mapper_sha256=digest(args.run_dir / "mapper.safetensors"),
        train_ids=[r["id"] for r in rows],
        heldout_ids=[r["id"] for r in dev],
        evaluation_scope=args.evaluation_scope,
        input_dim=features.shape[1] - 1,
        maximum_tokens=maximum,
        trainable_coefficients=sum(t.numel() for t in weights.values()),
        training_prefix_mse=float(((features @ coef) - target.flatten(1)).square().mean()),
        training_length_max_error=float(((features @ length_coef) - lengths).abs().max()),
        teacher_tensors_used_only_at_fit=True,
        qwen_weights_changed=False,
        scope=f"Continuous prefix-vector regression; {args.evaluation_scope}, awaiting evaluation",
    )
    (args.run_dir / "config.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info), flush=True)


def evaluate(args):
    info = json.loads((args.run_dir / "config.json").read_text())
    path = args.run_dir / "mapper.safetensors"
    if digest(path) != info["mapper_sha256"]:
        raise ValueError("Mapper checksum differs")
    weights = load_file(str(path))
    rows = json.loads(args.inputs.read_text())["rows"]
    for row in rows:
        if set(row) != {"id", "split", "features"}:
            raise ValueError("Numeric-only evaluation spec required")
        if row["id"] not in info["train_ids"] + info.get("heldout_ids", info.get("development_ids", [])):
            raise ValueError("Unknown ID")
    wrapper, builder, speaker = load_backbone(info)
    results = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    chosen = [(r["id"], r["split"], r["features"]) for r in rows if r["id"] in args.ids]
    chosen.append(("null", "control", [0.0] * info["input_dim"]))
    for label, split, values in chosen:
        prefix, length = predict_prefix(
            torch.tensor(values, dtype=torch.float32), weights["prefix_coef"], weights["length_coef"]
        )
        prefix = prefix.to(speaker.device)
        output = render(builder, wrapper, prefix, speaker, args.output_dir / f"{label}.wav", 105, 42)
        result = dict(
            id=label,
            split=split,
            predicted_prefix_length=length,
            human_listening_status="pending",
            **output,
        )
        results.append(result)
        print(json.dumps(result), flush=True)
    (args.output_dir / "results.json").write_text(
        json.dumps(
            {
                "teacher_files_read": False,
                "text_used": False,
                "evaluation_scope": info.get("evaluation_scope", "development"),
                "results": results,
                "mapper_sha256": info["mapper_sha256"],
                "inputs_sha256": digest(args.inputs),
            },
            indent=2,
        )
        + "\n"
    )


def main():
    if not Path("/.dockerenv").exists():
        raise RuntimeError("Use the managed worker")
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("fit")
    train.add_argument("--dataset", type=Path, required=True)
    train.add_argument(
        "--evaluation-scope", choices=["development", "fresh-test"], default="development"
    )
    test = sub.add_parser("evaluate")
    test.add_argument("--inputs", type=Path, required=True)
    test.add_argument("--ids", nargs="+", required=True)
    test.add_argument("--output-dir", type=Path, required=True)
    for cmd in [train, test]:
        cmd.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    (fit if args.command == "fit" else evaluate)(args)


if __name__ == "__main__":
    main()
