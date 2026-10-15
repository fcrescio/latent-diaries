"""Build a deterministic local-only inventory; do not print archive filenames."""
import argparse
import hashlib
import json
import random
from pathlib import Path

from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--paths-file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=256)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Preserve frozen inventories; use a new output file")
    if args.paths_file:
        paths = args.paths_file.read_text().splitlines()
    else:
        paths = [str(p) for p in sorted(args.root.rglob('*'))
                 if p.suffix.lower() in {'.jpg', '.jpeg', '.png'} and p.is_file()]
    random.Random(42).shuffle(paths)
    rows = []
    root = args.root.resolve()
    for value in paths:
        path = Path(value)
        if not path.resolve().is_relative_to(root):
            continue
        try:
            with Image.open(path) as image:
                if min(image.size) < 600:
                    continue
                exif = image.getexif()
                capture = exif.get(36867) or exif.get(306)
                image.load()
                size = list(image.size)
            rows.append({"id": f"p{len(rows):04d}", "path": str(path),
                         "group": hashlib.sha256(str(path.parent).encode()).hexdigest()[:12],
                         "capture": str(capture) if capture else None, "size": size})
        except (OSError, ValueError, Image.DecompressionBombError):
            continue
        if len(rows) >= args.limit:
            break
    args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    args.output.write_text(json.dumps(rows, indent=2) + '\n')
    args.output.chmod(0o600)
    print(f"Inventoried {len(rows)} images locally; filenames and metadata not printed")


if __name__ == "__main__":
    main()
