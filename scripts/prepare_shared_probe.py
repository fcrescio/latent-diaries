"""Create local training-only teacher caches and a separate numeric evaluation spec."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from latent_diaries.inversion import digest
from latent_diaries.shared_mapper import validate_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reuse-cache-root", type=Path)
    args = parser.parse_args()
    design = json.loads(args.design.read_text())
    validate_split(design["rows"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for row in design["rows"]:
        folder = args.output_dir / row["id"]
        reused = args.reuse_cache_root / row["id"] if args.reuse_cache_root else None
        if reused and (reused / "config.json").exists():
            folder = reused
        if not (folder / "config.json").exists():
            subprocess.run(
                [
                    sys.executable,
                    "scripts/cache_inversion_teacher.py",
                    "--output-dir",
                    str(folder),
                    "--text",
                    row["teacher_text"],
                ],
                check=True,
            )
        info = json.loads((folder / "config.json").read_text())
        import hashlib

        if info["training_text_sha256"] != hashlib.sha256(row["teacher_text"].encode()).hexdigest():
            raise ValueError("Existing cache belongs to a different sentence")
        entries.append(
            {k: row[k] for k in ["id", "split", "features"]}
            | {"cache_dir": str(folder), "cache_sha256": digest(folder / "teacher.safetensors")}
        )
    (args.output_dir / "dataset.json").write_text(
        json.dumps(
            {
                "design_sha256": digest(args.design),
                "rows": entries,
            },
            indent=2,
        )
        + "\n"
    )
    # Evaluation input deliberately contains no teacher paths, text or targets.
    (args.output_dir / "eval-inputs.json").write_text(
        json.dumps({"rows": [{k: r[k] for k in ["id", "split", "features"]} for r in entries]}, indent=2)
        + "\n"
    )


if __name__ == "__main__":
    main()
