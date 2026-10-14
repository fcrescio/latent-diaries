import json
from pathlib import Path
import pytest
from latent_diaries.review import ingest
from latent_diaries.demo import create_demo
from latent_diaries.storage import open_db


def test_ingest_idempotent(tmp_path):
    d = tmp_path / "demo"
    db, assets = tmp_path / "db", tmp_path / "assets"
    assert create_demo(d, db, assets) == 3
    assert ingest(d / "demo.jsonl", db, assets) == 3
    with open_db(db) as con:
        assert con.execute("SELECT COUNT(*) FROM clips").fetchone()[0] == 3


def test_reject_unsafe_id(tmp_path):
    manifest = tmp_path / "m.jsonl"
    manifest.write_text(json.dumps({"id": "../bad", "run_id": "x", "image": "x.png", "audio": "x.wav"}))
    with pytest.raises(ValueError, match="Unsafe clip id"):
        ingest(manifest, tmp_path / "db", tmp_path / "assets")
