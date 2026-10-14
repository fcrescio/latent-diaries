"""Import trusted local image/audio pairs into the private review gallery."""
from __future__ import annotations
import hashlib
import json
import os
import shutil
from pathlib import Path
from .storage import open_db

EXTENSIONS = {"image": {".jpg", ".jpeg", ".png", ".webp"},
              "audio": {".wav", ".mp3", ".ogg", ".flac"}}


def ingest(manifest: Path, db_path: Path, assets: Path) -> int:
    """JSONL with id, run_id, image, audio, blind_label. Paths relative to manifest directory."""
    assets.mkdir(parents=True, exist_ok=True)
    manifest = manifest.resolve()
    count = 0
    with open_db(db_path) as conn, manifest.open(encoding="utf8") as src:
        for line_no, line in enumerate(src, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            clip_id = str(row["id"])
            if not clip_id or not all(c.isalnum() or c in "-_." for c in clip_id):
                raise ValueError(f"Unsafe clip id on line {line_no}")
            paths = {}
            for kind in ("image", "audio"):
                path = (manifest.parent / row[kind]).resolve()
                if path.suffix.lower() not in EXTENSIONS[kind] or not path.is_file():
                    raise ValueError(f"Missing/unsupported {kind} on line {line_no}: {path}")
                digest = hashlib.sha256(path.read_bytes()).hexdigest()[:24]
                filename = digest + path.suffix.lower()
                dest = assets / filename
                if not dest.exists():
                    shutil.copyfile(path, dest)
                paths[kind] = filename
            conn.execute("""INSERT INTO clips(id,run_id,image_file,audio_file,blind_label)
                VALUES(?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET
                run_id=excluded.run_id,image_file=excluded.image_file,
                audio_file=excluded.audio_file,blind_label=excluded.blind_label""",
                (clip_id, str(row["run_id"]), paths["image"], paths["audio"],
                 str(row.get("blind_label", ""))))
            count += 1
    return count
