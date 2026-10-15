"""Freeze a local pilot split by capture day and near-duplicate components.

Private inventories/labels/manifests remain ignored. This is dataset preparation,
not ML training. Pixels and labels never enter the public protocol file.
"""
import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageOps


def dhash(image):
    small = image.convert("L").resize((9, 8))
    values = list(small.getdata())
    return sum((values[y * 9 + x] > values[y * 9 + x + 1]) << (y * 8 + x)
               for y in range(8) for x in range(8))


def grouped_split(rows, seed=42, heldout_target=12):
    """Union capture-day groups and perceptual neighbors before assigning splits."""
    parents = list(range(len(rows)))

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    for i, row in enumerate(rows):
        for j in range(i):
            other = rows[j]
            if row["capture_group"] == other["capture_group"] or (row["dhash"] ^ other["dhash"]).bit_count() <= 6:
                parents[root(i)] = root(j)
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[root(i)].append(row)
    components = list(groups.values())
    random.Random(seed).shuffle(components)
    held = 0
    result = []
    for group in components:
        choose = held < heldout_target and held + len(group) <= len(rows) - 20
        split = "heldout" if choose else "train"
        held += len(group) if choose else 0
        # Component identities are local pseudonyms, not dates or archive names.
        scene_group = hashlib.sha256("|".join(sorted(r["id"] for r in group)).encode()).hexdigest()[:16]
        result.extend(row | {"split": split, "scene_group": scene_group} for row in group)
    if held < 4 or len(rows) - held < 20:
        raise ValueError("Insufficient independent groups for the pilot")
    return sorted(result, key=lambda row: row["id"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=72)
    parser.add_argument("--exclusions", type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError("Use a fresh frozen split directory")
    labels = {row["id"]: row["label"] for row in
              [json.loads(line) for line in args.labels.read_text().splitlines() if line]}
    rows, fingerprints = [], set()
    excluded = set(json.loads(args.exclusions.read_text())) if args.exclusions else set()
    for row in json.loads(args.inventory.read_text()):
        label = labels.get(row["id"])
        if not label or not label["keep"] or row["id"] in excluded:
            continue
        if any(word in label["scene"].lower() for word in
               ["children", "child ", "people", "person", "competition", "selfie", "portrait"]):
            continue
        path = Path(row["path"])
        fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()
        if fingerprint in fingerprints:
            continue
        fingerprints.add(fingerprint)
        with Image.open(path) as original:
            image = ImageOps.exif_transpose(original)
            perceptual = dhash(image)
        capture_group = row["capture"][:10] if row["capture"] else row["group"]
        rows.append(row | {"capture_group": capture_group, "dhash": perceptual,
                           "teacher_text": label["narration_it"], "scene_label": label["scene"],
                           "image_sha256": fingerprint})
        if len(rows) >= args.max_images:
            break
    selected = grouped_split(rows)
    args.output_dir.mkdir(parents=True, mode=0o700)
    (args.output_dir / "design.json").write_text(json.dumps({"rows": selected}, indent=2) + "\n")
    manifest = [{"id": row["id"], "image": str(Path('/private/photos') / Path(row["path"]).relative_to(args.archive_root)),
                 "scene_group": row["scene_group"], "source": "user-authorized local archive",
                 "license": "Local tests/training only; no publication or upload authorized"} for row in selected]
    (args.output_dir / "images.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for file in args.output_dir.iterdir():
        file.chmod(0o600)
    print(json.dumps({"selected": len(selected), "train": sum(r['split'] == 'train' for r in selected),
                      "heldout": sum(r['split'] == 'heldout' for r in selected),
                      "independent_groups": len({r['scene_group'] for r in selected})}))


if __name__ == "__main__":
    main()
