"""Make a private, audio-only blinded listening page; no images or quality scores."""

import argparse
import html
import json
import random
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--first-label", default="A")
    parser.add_argument("--label-prefix", default="")
    parser.add_argument("--shuffle-seed", type=int, default=20261014)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Use a fresh listening directory")
    results = json.loads(args.results.read_text())["results"]
    # Reused null audio is represented only once, never as multiple observations.
    unique = {row["audio_sha256"]: row for row in results if row.get("audio")}
    rows = list(unique.values())
    if any(not (c.isascii() and (c.isalnum() or c in "-_")) for c in args.label_prefix):
        raise ValueError("Label prefix must contain only ASCII letters, digits, hyphens or underscores")
    if (
        len(args.first_label) != 1
        or not "A" <= args.first_label <= "Z"
        or ord(args.first_label) + len(rows) - 1 > ord("Z")
    ):
        raise ValueError("Choose an uppercase first label with room for all clips")
    random.Random(args.shuffle_seed).shuffle(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mapping = []
    sections = []
    for index, row in enumerate(rows):
        label = args.label_prefix + chr(ord(args.first_label) + index)
        filename = f"clip-{label}.wav"
        shutil.copyfile(row["audio"], args.output_dir / filename)
        mapping.append({"label": label, "source": row})
        sections.append(
            f"<section><h2>Clip {html.escape(label)}</h2>"
            f'<audio controls preload="none" src="{filename}"></audio></section>'
        )
    page = (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>Private audio listening probe</title>"
        "<style>body{font:18px system-ui;max-width:700px;margin:2rem auto;padding:1rem}"
        "section{margin:2rem 0}audio{width:100%}</style>"
        "<h1>Audio listening probe</h1><p>For each clip, report whether the Italian "
        "is clear, partly intelligible or mostly unintelligible. Note any repeats "
        "or unclear words. This is an audio-only research comparison; no image "
        "grounding assessment is requested. No ratings are generated automatically.</p>"
        + "".join(sections)
        + "</html>"
    )
    (args.output_dir / "index.html").write_text(page)
    (args.output_dir / "private-mapping.json").write_text(json.dumps(mapping, indent=2) + "\n")
    (args.output_dir / "private-mapping.json").chmod(0o600)
    print(json.dumps({"clips": len(mapping), "page": str(args.output_dir / "index.html")}))


if __name__ == "__main__":
    main()
