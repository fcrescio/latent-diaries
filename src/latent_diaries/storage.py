"""Shared SQLite schema for the review app and import jobs."""
from __future__ import annotations
import sqlite3
from pathlib import Path

def open_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path, timeout=10)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA foreign_keys=ON")
    con.executescript("""
        CREATE TABLE IF NOT EXISTS clips (
            id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            image_file TEXT NOT NULL,
            audio_file TEXT NOT NULL,
            blind_label TEXT NOT NULL DEFAULT '',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS ratings (
            clip_id TEXT NOT NULL REFERENCES clips(id),
            reviewer TEXT NOT NULL,
            grounding INTEGER NOT NULL CHECK (grounding BETWEEN 1 AND 5),
            voice INTEGER NOT NULL CHECK (voice BETWEEN 1 AND 5),
            evocation INTEGER NOT NULL CHECK (evocation BETWEEN 1 AND 5),
            surprise INTEGER NOT NULL CHECK (surprise BETWEEN 1 AND 5),
            note TEXT NOT NULL DEFAULT '',
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (clip_id, reviewer)
        );
    """)
    return con

