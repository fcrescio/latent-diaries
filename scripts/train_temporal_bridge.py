"""Bounded T0/T1 continuous-prefix and length diagnostic, backbones frozen.

This first bridge check uses offline projected supervision, not an acoustic
optimization claim. Training controls must speak before extending the pilot.
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
from image_probe import save
from safetensors.torch import load_file

from latent_diaries.inversion import digest
from latent_diaries.temporal_reader import TemporalReader, prefix_objective


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--targets", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--image-only", action="store_true")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--schedule-steps", type=int, default=2000)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if not Path("/.dockerenv").exists() or not 1 <= args.steps <= args.schedule_steps <= 3000:
        raise ValueError("Managed worker and at most3000 prespecified steps required")
    if args.run_dir.exists() and not args.resume:
        raise ValueError("Fresh run or explicit provenance-checked resume required")
    data = json.loads(args.targets.read_text())
    rows = data["records"]
    training = [r for r in rows if r["split"] == "train"]
    if len(training) != 16 or len(rows) != 24 or {r["split"] for r in rows} != {"train", "validation"}:
        raise ValueError("Frozen16/8-sequence diagnostic split required")
    if {r["scene_group"] for r in training} & {
        r["scene_group"] for r in rows if r["split"] == "validation"
    }:
        raise ValueError("Capture-group leakage")
    # No validation target is loaded or used to size/normalize/optimize the reader.
    targets = []
    for row in training:
        if digest(Path(row["prefix"])) != row["prefix_sha256"]:
            raise ValueError("Training prefix changed")
        targets.append(load_file(row["prefix"])["soft_tokens"][0])
    maximum = max(len(t) for t in targets)
    values = torch.cat(targets)
    mean = values.mean(0)
    scale = values.std(0, unbiased=False).clamp_min(1e-4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    torch.set_num_threads(4)
    reader = TemporalReader(maximum, mean, scale, feedback=not args.image_only, width=128).to(device)
    optimizer = torch.optim.AdamW(reader.parameters(), lr=0.001, weight_decay=0)
    target = torch.zeros(16, maximum, 1024, device=device)
    lengths = torch.tensor([len(t) for t in targets], device=device)
    mask = torch.arange(maximum, device=device)[None] < lengths[:, None]
    for i, t in enumerate(targets):
        target[i, : len(t)] = t.to(device)
    image = torch.tensor([r["image"] for r in training], device=device)
    audio = torch.tensor([r["audio"] for r in training], device=device)
    present = torch.ones(16, dtype=torch.bool, device=device)
    info = {k: data[k] for k in ["model_id", "model_revision", "role_ids", "lock_sha256"]}
    if info["lock_sha256"] != digest(Path("uv.lock")):
        raise ValueError("Worker lock changed")
    info.update(
        targets_sha256=digest(args.targets),
        source_sha256=digest(Path(__file__)),
        reader_source_sha256=digest(Path("src/latent_diaries/temporal_reader.py")),
        source_revision=os.environ.get("LD_CODE_REVISION"),
        feedback=reader.feedback,
        maximum_tokens=maximum,
        width=128,
        seed=42,
        learning_rate=0.001,
        schedule_steps=args.schedule_steps,
        trainable_parameters=sum(p.numel() for p in reader.parameters()),
        train_ids=[r["id"] for r in training],
        validation_ids=[r["id"] for r in rows if r["split"] == "validation"],
        objective="Training-only masked normalized projected-prefix MSE plus normalized length MSE",
        acoustic_optimization_exercised=False,
        backbones_frozen=True,
        scope="Teacher-history continuation bridge diagnostic; no first-fragment generator trained",
    )
    start = 0
    if args.resume:
        old = json.loads((args.run_dir / "config.json").read_text())
        for key, value in info.items():
            if key != "source_revision" and old[key] != value:
                raise ValueError("Resume provenance differs: " + key)
        if digest(args.run_dir / "reader.safetensors") != old["reader_sha256"]:
            raise ValueError("Checkpoint changed")
        reader.load_state_dict(load_file(str(args.run_dir / "reader.safetensors")))
        state = torch.load(args.run_dir / "state.pt", map_location=device, weights_only=True)
        if state["step"] != old["completed_steps"] or state["reader_sha256"] != old["reader_sha256"]:
            raise ValueError("Checkpoint/optimizer pair inconsistent")
        optimizer.load_state_dict(state["optimizer"])
        start = state["step"]
        info = old
        torch.set_rng_state(state["cpu_rng"].cpu())
        if state["cuda_rng"]:
            torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda_rng"]])
    args.run_dir.mkdir(mode=0o700, exist_ok=args.resume)
    started = time.monotonic()
    with (args.run_dir / "loss.jsonl").open("a") as log:
        for step in range(start, args.steps):
            lr = 0.001 * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / args.schedule_steps)))
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            prefix, _, _, length = reader(image, audio, present)
            loss, reconstruction, length_loss = prefix_objective(
                prefix, target, mask, reader.scale, length, lengths
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite bridge loss; diagnose before continuing")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(reader.parameters(), 1.0)
            if not torch.isfinite(norm) or norm <= 0:
                raise RuntimeError("Missing/nonfinite reader gradients")
            optimizer.step()
            if (step + 1) % 100 == 0 or step + 1 == args.steps:
                metric = {
                    "step": step + 1,
                    "loss": float(loss.detach()),
                    "prefix_mse": float(reconstruction.detach()),
                    "length_mse": float(length_loss.detach()),
                    "gradient_norm": float(norm),
                    "elapsed_seconds": time.monotonic() - started,
                }
                log.write(json.dumps(metric) + "\n")
                log.flush()
                print(json.dumps(metric), flush=True)
            if (step + 1) % 500 == 0 or step + 1 == args.steps:
                info["peak_allocated_vram_bytes"] = (
                    torch.cuda.max_memory_allocated() if device == "cuda" else None
                )
                save(args.run_dir, reader, optimizer, step + 1, info)


if __name__ == "__main__":
    main()
