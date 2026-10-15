"""Global image-embedding reader baseline; evaluation accepts vectors, never labels/text."""

import argparse
import json
import math
import os
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from latent_diaries.image_query_reader import ImageQueryReader
from latent_diaries.image_reader import GlobalImageReader, validate_image_inputs
from latent_diaries.inversion import digest, load_backbone, render
from latent_diaries.sequence_reader import masked_prefix_loss
from latent_diaries.shared_mapper import validate_split


def make_reader(info, weights):
    reader_type = ImageQueryReader if info.get("architecture") == "image-query" else GlobalImageReader
    reader = reader_type(
        info["maximum_tokens"], weights["mean"], weights["scale"],
        weights["length_coef"], info["width"],
    )
    reader.load_state_dict(weights)
    return reader


def save(folder, reader, optimizer, step, info):
    save_file(
        {k: v.detach().cpu().contiguous() for k, v in reader.state_dict().items()},
        str(folder / "reader.tmp.safetensors"),
    )
    checksum = digest(folder / "reader.tmp.safetensors")
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "step": step,
            "reader_sha256": checksum,
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        },
        folder / "state.tmp.pt",
    )
    (folder / "reader.tmp.safetensors").replace(folder / "reader.safetensors")
    (folder / "state.tmp.pt").replace(folder / "state.pt")
    info.update(completed_steps=step, reader_sha256=checksum)
    (folder / "config.tmp.json").write_text(json.dumps(info, indent=2) + "\n")
    (folder / "config.tmp.json").replace(folder / "config.json")


def train(args):
    dataset = json.loads(args.dataset.read_text())
    rows, tests = validate_split(dataset["rows"])
    if any(len(row["features"]) != 768 for row in dataset["rows"]):
        raise ValueError("Expected frozen 768D image vectors")
    if {r["scene_group"] for r in rows} & {r["scene_group"] for r in tests}:
        raise ValueError("Scene-group leakage")
    if args.steps < 1 or args.steps > args.schedule_steps:
        raise ValueError("Steps must lie in the fixed learning-rate schedule")
    folder = args.run_dir
    if not args.resume and folder.exists() and any(folder.iterdir()):
        raise ValueError("Use a fresh directory or --resume")
    torch.manual_seed(42)
    prefixes, metadata, hashes = [], [], {}
    for row in rows:
        path = Path(row["cache_dir"])
        metadata.append(json.loads((path / "config.json").read_text()))
        prefixes.append(load_file(str(path / "prefix.safetensors"))["soft_tokens"][0])
        hashes[row["id"]] = digest(path / "prefix.safetensors")
    info = {k: metadata[0][k] for k in ["model_id", "model_revision", "lock_sha256"]}
    if info["lock_sha256"] != digest(Path("uv.lock")):
        raise ValueError("Dependency lock differs")
    for value in metadata:
        for key in ["model_id", "model_revision", "lock_sha256", "role_ids"]:
            if value[key] != metadata[0][key]:
                raise ValueError("Teacher provenance differs")
    maximum = max(len(p) for p in prefixes)
    target = torch.zeros(len(rows), maximum, prefixes[0].shape[-1])
    lengths = torch.tensor([len(p) for p in prefixes])
    mask = torch.arange(maximum)[None] < lengths[:, None]
    for i, p in enumerate(prefixes):
        target[i, : len(p)] = p
    features = torch.tensor([r["features"] for r in rows], dtype=torch.float32)
    design = torch.cat([torch.ones(len(rows), 1), features], -1).double()
    if args.fixed_length:
        if not torch.equal(lengths, lengths[:1].expand_as(lengths)):
            raise ValueError('Fixed-length supervision requires identical teacher prefix lengths')
        length_coef = torch.zeros(features.shape[1] + 1)
        length_coef[0] = int(lengths[0])
    else:
        length_coef = torch.linalg.lstsq(design, lengths.double()[:, None]).solution[:, 0].float()
    mean = target[mask].mean(0)
    scale = target[mask].std(0, unbiased=False).clamp_min(1e-4)
    info.update(
        dataset_sha256=digest(args.dataset),
        design_sha256=dataset["design_sha256"],
        source_revision=os.environ.get("LD_CODE_REVISION"),
        source_sha256=digest(Path(__file__)),
        reader_source_sha256=digest(Path("src/latent_diaries/image_query_reader.py" if args.architecture == "image-query" else "src/latent_diaries/image_reader.py")),
        training_prefix_sha256=hashes,
        train_ids=[r["id"] for r in rows],
        heldout_ids=[r["id"] for r in tests],
        input_dim=768,
        architecture=args.architecture,
        fixed_length=args.fixed_length,
        maximum_tokens=maximum,
        width=args.width,
        seed=42,
        learning_rate=0.001,
        schedule_steps=args.schedule_steps,
        warmup_steps=100,
        evaluation_scope=args.evaluation_scope or ("exposed-development" if args.architecture == "image-query" else "fresh-test"),
        qwen_weights_changed=False,
        scope=("Position-sharing global-image query decoder; exposed development" if args.architecture == "image-query" else "Global normalized image-vector MLP; cached teacher-prefix reconstruction"),
        image_features_sha256=dataset["image_features_sha256"],
        vision_manifest_sha256=dataset["vision_manifest_sha256"],
    )
    if args.resume:
        saved = json.loads((folder / "config.json").read_text())
        for key, value in info.items():
            if key != "source_revision" and saved[key] != value:
                raise ValueError(f"Resume configuration differs: {key}")
        info = saved
    folder.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    reader_type = ImageQueryReader if args.architecture == "image-query" else GlobalImageReader
    reader = reader_type(maximum, mean, scale, length_coef, width=args.width).to(device)
    optimizer = torch.optim.AdamW(reader.parameters(), lr=info["learning_rate"], weight_decay=0.0)
    start = 0
    if args.resume:
        if digest(folder / "reader.safetensors") != info["reader_sha256"]:
            raise ValueError("Reader checksum differs")
        reader.load_state_dict(load_file(str(folder / "reader.safetensors")))
        state = torch.load(folder / "state.pt", weights_only=True, map_location=device)
        if state["reader_sha256"] != info["reader_sha256"] or state["step"] != info["completed_steps"]:
            raise ValueError("Checkpoint pair inconsistent")
        optimizer.load_state_dict(state["optimizer"])
        start = state["step"]
        torch.set_rng_state(state["cpu_rng"].cpu())
        if state["cuda_rng"]:
            torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda_rng"]])
    target, features, mask, lengths = [v.to(device) for v in [target, features, mask, lengths]]
    reader.train()
    info["trainable_parameters"] = sum(p.numel() for p in reader.parameters())
    with (folder / "loss.jsonl").open("a") as log:
        for step in range(start, args.steps):
            learning_rate = info["learning_rate"] * min(1.0, (step + 1) / info["warmup_steps"])
            learning_rate *= 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / args.schedule_steps))
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            optimizer.zero_grad(set_to_none=True)
            predicted, predicted_mask, predicted_lengths = reader(features)
            if args.architecture == "global-mlp" and (not torch.equal(predicted_mask, mask) or not torch.equal(predicted_lengths, lengths)):
                raise ValueError("Numeric length prediction differs from training masks")
            loss = masked_prefix_loss(predicted, target, mask, reader.scale)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite reconstruction loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(reader.parameters(), 1.0)
            if not torch.isfinite(norm) or norm <= 0:
                raise RuntimeError("Missing/nonfinite reader gradients")
            optimizer.step()
            metric = {
                "step": step + 1,
                "normalized_masked_mse": float(loss.detach()),
                "learning_rate": learning_rate,
                "gradient_norm": float(norm),
            }
            log.write(json.dumps(metric) + "\n")
            log.flush()
            if (step + 1) % 100 == 0 or step == start:
                print(json.dumps(metric), flush=True)
            if (step + 1) % 400 == 0 or step + 1 == args.steps:
                info["peak_allocated_vram_bytes"] = (
                    torch.cuda.max_memory_allocated() if device.type == "cuda" else None
                )
                save(folder, reader, optimizer, step + 1, info)


def evaluate(args):
    info = json.loads((args.run_dir / "config.json").read_text())
    path = args.run_dir / "reader.safetensors"
    if digest(path) != info["reader_sha256"]:
        raise ValueError("Reader checksum differs")
    reader = make_reader(info, load_file(str(path)))
    inputs = json.loads(args.inputs.read_text())
    validate_image_inputs(inputs["rows"], info["train_ids"] + info["heldout_ids"], args.allow_new_images)
    wrapper, builder, speaker = load_backbone(info)
    reader = reader.to(speaker.device).eval()
    chosen = [(r["id"], r["split"], r["features"]) for r in inputs["rows"] if r["id"] in args.ids]
    chosen.append(("null", "control", [0.0] * info["input_dim"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for label, split, values in chosen:
        with torch.no_grad():
            prefix, mask, lengths = reader(torch.tensor([values], device=speaker.device, dtype=torch.float32))
        length = int(lengths[0])
        assert bool(mask[0, :length].all())
        output = render(
            builder, wrapper, prefix[:, :length], speaker, args.output_dir / f"{label}.wav", args.max_frames, 42
        )
        result = {
            "id": label,
            "split": split,
            "predicted_prefix_length": length,
            "human_listening_status": "pending",
            **output,
        }
        results.append(result)
        print(json.dumps(result), flush=True)
    (args.output_dir / "results.json").write_text(
        json.dumps(
            {
                "reader_sha256": info["reader_sha256"],
                "inputs_sha256": digest(args.inputs),
                "teacher_files_read": False,
                "image_conditioning": True,
                "new_image_ids_allowed": args.allow_new_images,
                "linguistic_labels_read": False,
                "text_used": False,
                "results": results,
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
    trainer = sub.add_parser("train")
    trainer.add_argument("--dataset", type=Path, required=True)
    trainer.add_argument("--steps", type=int, required=True)
    trainer.add_argument("--schedule-steps", type=int, default=1600)
    trainer.add_argument("--resume", action="store_true")
    trainer.add_argument("--width", type=int, choices=[128, 256], default=128)
    trainer.add_argument("--fixed-length", action="store_true")
    trainer.add_argument("--evaluation-scope", choices=["fresh-test", "exposed-development", "development"])
    trainer.add_argument("--architecture", choices=["global-mlp", "image-query"], default="global-mlp")
    evaluator = sub.add_parser("evaluate")
    evaluator.add_argument("--max-frames", type=int, default=105)
    evaluator.add_argument("--allow-new-images", action="store_true")
    evaluator.add_argument("--inputs", type=Path, required=True)
    evaluator.add_argument("--ids", nargs="+", required=True)
    evaluator.add_argument("--output-dir", type=Path, required=True)
    for cmd in [trainer, evaluator]:
        cmd.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    (train if args.command == "train" else evaluate)(args)


if __name__ == "__main__":
    main()
