"""Train shared numeric-latent mapper, or evaluate without text/teacher files.

This is an authored-factor compositional probe, not an image reader. Held-out
teacher caches are never loaded by training. Inference takes numeric inputs only.
"""

import argparse
import json
import os
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from latent_diaries.inversion import (
    digest,
    distribution_kl,
    load_backbone,
    render,
    trajectory_logits,
)
from latent_diaries.shared_mapper import SharedMapper, validate_split


def save(folder, mapper, optimizer, step, metadata):
    save_file(
        {k: v.detach().cpu().contiguous() for k, v in mapper.state_dict().items()},
        str(folder / "mapper.tmp.safetensors"),
    )
    checksum = digest(folder / "mapper.tmp.safetensors")
    torch.save(
        {"optimizer": optimizer.state_dict(), "step": step, "mapper_sha256": checksum},
        folder / "state.tmp.pt",
    )
    (folder / "mapper.tmp.safetensors").replace(folder / "mapper.safetensors")
    (folder / "state.tmp.pt").replace(folder / "state.pt")
    metadata.update(completed_steps=step, mapper_sha256=checksum)
    (folder / "config.tmp.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (folder / "config.tmp.json").replace(folder / "config.json")


def train(args):
    dataset = json.loads(args.dataset.read_text())
    rows, heldout = validate_split(dataset["rows"])
    if args.steps < 1 or args.learning_rate <= 0:
        raise ValueError("Positive steps and learning rate required")
    folder = args.run_dir
    baseline = json.loads(Path("artifacts/m0_italian.json").read_text())
    metadata = {k: baseline[k] for k in ["model_id", "model_revision"]}
    metadata.update(
        dataset_sha256=digest(args.dataset),
        source_sha256=digest(Path(__file__)),
        mapper_source_sha256=digest(Path("src/latent_diaries/shared_mapper.py")),
        conditioning_source_sha256=digest(Path("src/latent_diaries/conditioning.py")),
        inversion_source_sha256=digest(Path("src/latent_diaries/inversion.py")),
        lock_sha256=digest(Path("uv.lock")),
        source_revision=os.environ.get("LD_CODE_REVISION"),
        seed=42,
        tokens=32,
        input_dim=3,
        width=128,
        learning_rate=args.learning_rate,
        train_ids=[r["id"] for r in rows],
        heldout_ids=[r["id"] for r in heldout],
        scope="Authored numeric-factor probe, not image conditioning",
    )
    if args.resume:
        saved = json.loads((folder / "config.json").read_text())
        for k, v in metadata.items():
            if k != "source_revision" and saved[k] != v:
                raise ValueError(f"Resume configuration differs: {k}")
        metadata = saved
    elif folder.exists() and any(folder.iterdir()):
        raise ValueError("Use fresh run directory or --resume")
    folder.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    wrapper, builder, speaker = load_backbone(metadata)
    device = speaker.device
    with torch.no_grad():
        pad_id = torch.tensor([[wrapper.model.config.tts_pad_token_id]], device=device)
        pad = wrapper.model.talker.text_projection(wrapper.model.talker.get_text_embeddings()(pad_id))
    mapper = SharedMapper(pad.expand(1, 32, -1), width=128).to(device)
    optimizer = torch.optim.AdamW(mapper.parameters(), lr=args.learning_rate, weight_decay=0)
    start = 0
    if args.resume:
        if digest(folder / "mapper.safetensors") != metadata["mapper_sha256"]:
            raise ValueError("Mapper checksum differs")
        mapper.load_state_dict(load_file(str(folder / "mapper.safetensors")))
        state = torch.load(folder / "state.pt", weights_only=True, map_location=device)
        if (
            state["mapper_sha256"] != metadata["mapper_sha256"]
            or state["step"] != metadata["completed_steps"]
        ):
            raise ValueError("Inconsistent checkpoint pair")
        optimizer.load_state_dict(state["optimizer"])
        start = state["step"]
    caches = []
    # Intentionally iterate train rows only: heldout caches remain unopened.
    for row in rows:
        path = Path(row["cache_dir"])
        info = json.loads((path / "config.json").read_text())
        for key in ["model_id", "model_revision", "lock_sha256"]:
            if info[key] != metadata[key]:
                raise ValueError(f"Teacher provenance differs: {key}")
        if info["role_ids"] != list(builder.role_ids):
            raise ValueError("Teacher protocol differs")
        if digest(path / "teacher.safetensors") != row["cache_sha256"]:
            raise ValueError("Teacher cache checksum differs")
        cache = {k: v.to(device) for k, v in load_file(str(path / "teacher.safetensors")).items()}
        caches.append(cache)
    features = torch.tensor([r["features"] for r in rows], device=device, dtype=torch.float32)
    metadata["trainable_parameters"] = sum(p.numel() for p in mapper.parameters())
    with (folder / "loss.jsonl").open("a") as log:
        for step in range(start, args.steps):
            index = step % len(rows)
            optimizer.zero_grad(set_to_none=True)
            prefix = mapper(features[index : index + 1])
            cache = caches[index]
            first, residual = trajectory_logits(wrapper, builder, prefix, speaker, cache["audio_codes"])
            first_loss = distribution_kl(first, cache["codec0_logits"])
            residual_loss = distribution_kl(residual, cache["residual_logits"])
            loss = first_loss + residual_loss
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite loss; inspect before retry")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(mapper.parameters(), 1.0)
            if not torch.isfinite(norm) or norm <= 0:
                raise RuntimeError("Missing or nonfinite mapper gradient")
            optimizer.step()
            metrics = {
                "step": step + 1,
                "example_id": rows[index]["id"],
                "loss": float(loss.detach()),
                "codec0_kl": float(first_loss.detach()),
                "residual_kl": float(residual_loss.detach()),
            }
            log.write(json.dumps(metrics) + "\n")
            log.flush()
            if (step + 1) % 30 == 0 or step == start:
                print(json.dumps(metrics), flush=True)
            if (step + 1) % 300 == 0 or step + 1 == args.steps:
                assert all(p.grad is None for p in wrapper.model.parameters())
                metadata["frozen_model_gradients_absent"] = True
                metadata["peak_allocated_vram_bytes"] = torch.cuda.max_memory_allocated()
                save(folder, mapper, optimizer, step + 1, metadata)


def evaluate(args):
    folder = args.run_dir
    metadata = json.loads((folder / "config.json").read_text())
    if digest(folder / "mapper.safetensors") != metadata["mapper_sha256"]:
        raise ValueError("Mapper checksum differs")
    weights = load_file(str(folder / "mapper.safetensors"))
    mapper = SharedMapper(weights["base"], metadata["input_dim"], metadata["width"])
    mapper.load_state_dict(weights)
    wrapper, builder, speaker = load_backbone(metadata)
    mapper = mapper.to(speaker.device).eval()
    inputs = json.loads(args.inputs.read_text())
    # Reject teacher-bearing specs even though we never dereference such fields.
    for row in inputs["rows"]:
        if set(row) != {"id", "features", "split"}:
            raise ValueError("Evaluation spec accepts only IDs, numeric features and split")
        if row["id"] not in metadata["train_ids"] + metadata["heldout_ids"]:
            raise ValueError("Unknown evaluation ID")
    results = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    null_outputs = {}
    for row in inputs["rows"]:
        if row["id"] not in args.ids:
            continue
        features = torch.tensor([row["features"]], device=speaker.device, dtype=torch.float32)
        for name, value in [("conditioned", features), ("null", torch.zeros_like(features))]:
            with torch.no_grad():
                prefix = mapper(value)
            for greedy in [False, True]:
                label = f"{row['id']}-{name}-{'greedy0' if greedy else 'sampled'}"
                if name == "null" and greedy in null_outputs:
                    output = null_outputs[greedy]
                else:
                    output = render(
                        builder,
                        wrapper,
                        prefix,
                        speaker,
                        args.output_dir / f"{label}.wav",
                        105,
                        42,
                        codec0_greedy=greedy,
                    )
                    if name == "null":
                        null_outputs[greedy] = output
                result = dict(
                    id=row["id"],
                    split=row["split"],
                    condition=name,
                    human_listening_status="pending",
                    **output,
                )
                results.append(result)
                print(json.dumps(result), flush=True)
    (args.output_dir / "results.json").write_text(
        json.dumps(
            {
                "mapper_sha256": metadata["mapper_sha256"],
                "inputs_sha256": digest(args.inputs),
                "teacher_files_read": False,
                "text_used": False,
                "results": results,
            },
            indent=2,
        )
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    trainer = sub.add_parser("train")
    trainer.add_argument("--dataset", type=Path, required=True)
    trainer.add_argument("--steps", type=int, required=True)
    trainer.add_argument("--learning-rate", type=float, default=0.001)
    trainer.add_argument("--resume", action="store_true")
    evaluator = sub.add_parser("evaluate")
    evaluator.add_argument("--inputs", type=Path, required=True)
    evaluator.add_argument("--ids", nargs="+", required=True)
    evaluator.add_argument("--output-dir", type=Path, required=True)
    for cmd in [trainer, evaluator]:
        cmd.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    (train if args.command == "train" else evaluate)(args)


if __name__ == "__main__":
    main()
