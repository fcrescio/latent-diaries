"""Check required configuration without showing secret values."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
if not Path(".env").exists():
    raise SystemExit("Missing .env: run 'make init' and fill credentials")
for key in ("GH_TOKEN", "GITHUB_OWNER", "HF_TOKEN", "HF_NAMESPACE"):
    print(f"{key}: {'configured' if os.getenv(key) else 'MISSING'}")
print("Paid HF Jobs approval:", os.getenv("LD_APPROVE_PAID_JOBS") == "yes")
print("Review password:", "configured" if os.getenv("REVIEW_PASSWORD") else "MISSING")
