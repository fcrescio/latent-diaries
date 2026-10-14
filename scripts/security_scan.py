"""Minimal local guard; add GitHub secret scanning in production."""
from __future__ import annotations
import subprocess

tracked = subprocess.run(["git", "ls-files"], capture_output=True, text=True, check=False).stdout.splitlines()
for forbidden in (".env", "run-data/", "artifacts/", "checkpoints/", "logs/"):
    offenders = [x for x in tracked if x == forbidden or
                 (forbidden.endswith("/") and x.startswith(forbidden)) or
                 (forbidden == ".env" and x.startswith(".env.") and x != ".env.example")]
    if offenders:
        raise SystemExit(f"Sensitive path tracked in git: {offenders}")
print("No known secret/data paths tracked")
