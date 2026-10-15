"""Prepare local blinded image/audio review pairs; never publish source media."""
import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageOps

from latent_diaries.review import ingest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError("Use fresh listening directory")
    rows = {row["id"]: row for row in json.loads(args.design.read_text())["rows"]}
    plan = json.loads(args.plan.read_text())
    results = {row["id"]: row for row in json.loads(args.results.read_text())["results"]}
    ids = plan["heldout_ids"]
    assert all(rows[label]["split"] == "heldout" for label in ids)
    args.output_dir.mkdir(parents=True, mode=0o700)
    pairs = []
    for i, label in enumerate(ids):
        pairs.extend([
            {"image_id": label, "audio_id": label, "condition": "matched"},
            {"image_id": label, "audio_id": ids[plan["shuffle"][i]], "condition": "shuffled"},
            {"image_id": label, "audio_id": "null", "condition": "null"},
        ])
    pairs.append({"image_id": plan["training_id"], "audio_id": plan["training_id"], "condition": "train"})
    random.Random(2026101504).shuffle(pairs)
    images = {}
    for label in ids + [plan["training_id"]]:
        path = args.output_dir / f"{label}.jpg"
        with Image.open(rows[label]["path"]) as original:
            image = ImageOps.exif_transpose(original).convert("RGB")
            image.thumbnail((1024, 1024))
            image.save(path, quality=90)  # strip EXIF, do not copy originals
        images[label] = path.resolve()
    mapping, public_rows = [], []
    for i, pair in enumerate(pairs):
        blind = f"E4-{chr(65+i)}"
        row = {"id": f"m2p1-{chr(97+i)}", "run_id": "m2-private-pilot-001",
               "blind_label": blind, "image": str(images[pair["image_id"]]),
               "audio": str(Path(results[pair["audio_id"]]["audio"]).resolve())}
        public_rows.append(row)  # local gallery manifest only, NEVER an upload/export
        mapping.append(pair | {"label": blind, "id": row["id"],
                               "audio_sha256": results[pair["audio_id"]]["audio_sha256"]})
    manifest = args.output_dir / "review.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in public_rows))
    manifest.chmod(0o600)
    private_map = args.output_dir / "private-mapping.json"
    private_map.write_text(json.dumps(mapping, indent=2) + "\n")
    private_map.chmod(0o600)
    count = ingest(manifest, Path("run-data/reviews.sqlite3"), Path("run-data/review-assets"))
    print(f"Imported {count} genuinely generated private pairs; no ratings created")


if __name__ == "__main__":
    main()
