"""Training-only text projection; held-out teacher texts/prefixes are never loaded."""
import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from latent_diaries.inversion import digest, load_backbone


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not Path("/.dockerenv").exists():
        raise RuntimeError("Use managed Docker worker")
    if args.output_dir.exists():
        raise ValueError("Use fresh output directory")
    design = json.loads(args.design.read_text())
    manifest = json.loads((args.features_dir / "manifest.json").read_text())
    feature_file = args.features_dir / "features.safetensors"
    if digest(feature_file) != manifest["features_sha256"]:
        raise ValueError("Image feature checksum differs")
    vectors = load_file(str(feature_file))["embeddings"]
    by_id = {r["id"]: vectors[i] for i, r in enumerate(manifest["records"])}
    info = json.loads(Path("artifacts/m0_italian.json").read_text())
    wrapper, builder, _ = load_backbone(info)
    args.output_dir.mkdir(parents=True)
    entries = []
    for row in design["rows"]:
        folder = args.output_dir / row["id"]
        if row["split"] == "train":
            folder.mkdir()
            ids = wrapper._tokenize_texts([wrapper._build_assistant_text(row["teacher_text"])])[0]
            with torch.no_grad():
                prefix = wrapper.model.talker.text_projection(
                    wrapper.model.talker.get_text_embeddings()(ids[:, 3:-5]))
            save_file({"soft_tokens": prefix.cpu()}, str(folder / "prefix.safetensors"))
            provenance = {k: info[k] for k in ["model_id", "model_revision"]}
            provenance.update(lock_sha256=digest(Path("uv.lock")), role_ids=list(builder.role_ids))
            (folder / "config.json").write_text(json.dumps(provenance, indent=2) + "\n")
        entries.append({"id": row["id"], "split": row["split"], "scene_group": row["scene_group"],
                        "features": by_id[row["id"]].tolist(), "cache_dir": str(folder)})
    metadata = {"rows": entries, "design_sha256": digest(args.design),
                "image_features_sha256": digest(feature_file),
                "vision_manifest_sha256": digest(args.features_dir / "manifest.json"),
                "supervision": "training-only projected text vectors; no acoustic CE in this stage"}
    (args.output_dir / "dataset.json").write_text(json.dumps(metadata, indent=2) + "\n")
    evaluation = {"rows": [{k: row[k] for k in ["id", "split", "features"]} for row in entries]}
    (args.output_dir / "eval-inputs.json").write_text(json.dumps(evaluation, indent=2) + "\n")
    print(f"Prepared {len(entries)} image vectors; only training text was projected", flush=True)


if __name__ == "__main__":
    main()
