"""Ingest only a completed, checksum-verified frozen grid into local audio review."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from latent_diaries.omnivoice_protocol import get_plan, validate_reuse
from latent_diaries.storage import open_db
from services.review_web.app import create_app


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outputs", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--db", type=Path, default=Path("run-data/reviews.sqlite3"))
    p.add_argument("--assets", type=Path, default=Path("run-data/review-assets"))
    p.add_argument("--mapping", type=Path, required=True)
    a = p.parse_args()
    full = json.loads((a.outputs / "full-summary.json").read_text())
    if full.get("technical_pass") is not True or len(full["cases"]) != 20:
        raise RuntimeError("Complete actual frozen grid required")
    run_sha = digest(a.outputs / "run-provenance.json")
    plan = get_plan("full")
    plan = sorted(
        plan, key=lambda item: hashlib.sha256(("OMX-01/20261017/" + str(item)).encode()).hexdigest()
    )
    prepared = []
    for i, (mode, seed, seconds, _) in enumerate(plan, 1):
        case = f"{mode}_s{seed}_d{int(seconds)}"
        wav = a.outputs / (case + ".wav")
        codes = a.outputs / (case + ".safetensors")
        record = json.loads((a.outputs / (case + ".json")).read_text())
        validate_reuse(record, run_sha, mode, seed, seconds, digest(wav), digest(codes))
        prepared.append(
            {
                "id": f"omx-20261017-{i:02d}",
                "label": f"OMX-{i:02d}",
                "case": case,
                "wav_sha256": digest(wav),
                "source": wav,
                "voice_applies": mode in {"positive", "voice-empty"},
            }
        )
    if len({c["wav_sha256"] for c in prepared}) != 20:
        raise RuntimeError("Duplicate audio must not count as independent generations")
    if a.mapping.exists():
        raise RuntimeError("Preserve existing mapping; do not reshuffle")
    a.assets.mkdir(parents=True, exist_ok=True)
    target = a.assets / "omx-20261017"
    target.mkdir(mode=0o700, exist_ok=False)
    shutil.copyfile(a.reference, target / "reference.wav")
    # Initialize additive schema, preserving all historical image/audio ratings.
    create_app(str(a.db), str(a.assets), review_password="")
    with open_db(a.db) as con:
        for c in prepared:
            shutil.copyfile(c["source"], target / (c["id"] + ".wav"))
            con.execute(
                "INSERT INTO audio_probe_clips VALUES(?,?,?,?,?,?)",
                (
                    c["id"],
                    "OMX-01",
                    c["label"],
                    f"omx-20261017/{c['id']}.wav",
                    "omx-20261017/reference.wav",
                    int(c["voice_applies"]),
                ),
            )
    a.mapping.write_text(
        json.dumps([{k: v for k, v in c.items() if k != "source"} for c in prepared], indent=2) + "\n"
    )
    a.mapping.chmod(0o600)
    print(json.dumps({"ingested": 20, "ratings_created": 0, "review_path": "/audio-probe"}))


if __name__ == "__main__":
    main()
