"""Command entry point. ML training stages fail closed until implemented and tested."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def doctor() -> None:
    import platform
    from importlib.util import find_spec
    print("Python:", platform.python_version())
    print("HF credentials configured:", bool(os.getenv("HF_TOKEN")))
    print("Github credentials configured:", bool(os.getenv("GH_TOKEN")))
    for module in ("torch", "transformers", "qwen_tts", "huggingface_hub"):
        print(module, "installed" if find_spec(module) else "missing")
    if find_spec("torch"):
        import torch
        print("CUDA available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("Device:", torch.cuda.get_device_name())
    print("NOTE: doctor verifies installation only; does not validate textless inference.")


def not_ready(stage: str) -> None:
    raise SystemExit(f"NOT IMPLEMENTED: {stage}. Read AGENTS.md and docs/MILESTONES.md. "
                     "Never submit a paid cloud job for an unimplemented trainer.")


def main():
    parser = argparse.ArgumentParser(prog="latent-diaries")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor")
    sub.add_parser("demo-data")
    ingest = sub.add_parser("ingest-review")
    ingest.add_argument("--manifest", type=Path, required=True)
    qwen = sub.add_parser("qwen-baseline")
    qwen.add_argument("--output", type=Path, default=Path("artifacts/m0_italian.wav"))
    qwen.add_argument("--text", default=None)
    inversion = sub.add_parser("train-soft-prefix")
    inversion.add_argument("--target-audio", type=Path, default=Path("artifacts/m0_italian.wav"))
    inversion.add_argument("--run-dir", type=Path, default=Path("artifacts/m1-inversion"))
    inversion.add_argument("--steps", type=int, default=200)
    inversion.add_argument("--tokens", type=int, default=16)
    inversion.add_argument("--learning-rate", type=float, default=.03)
    inversion.add_argument("--seed", type=int, default=42)
    inversion.add_argument("--resume", action="store_true")
    inversion.add_argument("--residual-loss-weight", type=float, default=0.)
    inversion.add_argument("--teacher-cache-dir", type=Path)
    inversion.add_argument("--cached-gold-ce", action="store_true")
    inversion.add_argument("--init-prefix-dir", type=Path)
    inversion.add_argument("--rollout-every", type=int, default=0)
    inversion.add_argument("--rollout-weight", type=float, default=1.)
    prefix = sub.add_parser("infer-soft-prefix")
    prefix.add_argument("--run-dir", type=Path, required=True)
    prefix.add_argument("--output", type=Path, required=True)
    prefix.add_argument("--max-frames", type=int, default=120)
    prefix.add_argument("--seed", type=int, default=42)
    prefix.add_argument("--codec0-greedy", action="store_true",
                        help="Greedy first codebook; keep residual codebooks sampled")
    for name in ("train-reader", "narrate-image", "evaluate-grounding"):
        sub.add_parser(name)
    args = parser.parse_args()
    db_path = Path(os.environ.get("REVIEW_DB", "run-data/reviews.sqlite3"))
    assets = Path(os.environ.get("REVIEW_ASSETS", "run-data/review-assets"))
    if args.command == "doctor":
        doctor()
    elif args.command == "demo-data":
        from .demo import create_demo
        count = create_demo(Path("run-data/demo"), db_path, assets)
        print(f"Imported {count} SYNTHETIC NON-ML demonstration records")
    elif args.command == "ingest-review":
        from .review import ingest as ingest_reviews
        print("Imported", ingest_reviews(args.manifest, db_path, assets), "clip(s)")
    elif args.command == "qwen-baseline":
        from .qwen_baseline import DEFAULT_TEXT, run
        run(args.output, text=args.text or DEFAULT_TEXT)
    elif args.command == "train-soft-prefix":
        from .inversion import train
        train(args.target_audio, args.run_dir, args.steps, args.tokens, args.learning_rate,
              args.seed, args.resume, args.residual_loss_weight, args.teacher_cache_dir,
              args.cached_gold_ce, args.init_prefix_dir, args.rollout_every, args.rollout_weight)
    elif args.command == "infer-soft-prefix":
        from .inversion import infer
        infer(args.run_dir, args.output, args.max_frames, args.seed, args.codec0_greedy)
    else:
        not_ready(args.command)

if __name__ == "__main__":
    main()
