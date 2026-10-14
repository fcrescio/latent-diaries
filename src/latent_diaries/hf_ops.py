"""Explicit, private-by-default checkpoint and dataset publication to Hugging Face Hub."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv()


def repo_id(kind: str) -> str:
    namespace = os.getenv("HF_NAMESPACE", "").strip()
    repo = os.getenv("HF_MODEL_REPO" if kind == "model" else "HF_DATASET_REPO", "").strip()
    if not namespace or not repo:
        raise RuntimeError("Specify HF_NAMESPACE and HF_MODEL_REPO / HF_DATASET_REPO in .env")
    return repo if "/" in repo else f"{namespace}/{repo}"


def ensure_private_repo(api: HfApi, kind: str) -> str:
    if os.getenv("HF_PRIVATE", "true").lower() != "true":
        raise RuntimeError("Public repos need explicit code review and HUMAN_APPROVAL.md sign-off.")
    name = repo_id(kind)
    api.create_repo(repo_id=name, repo_type=kind, private=True, exist_ok=True)
    info = api.repo_info(repo_id=name, repo_type=kind)
    if not info.private:
        raise RuntimeError(f"REFUSING upload: {name} is PUBLIC; use a private repo")
    return name


def publish_model(api: HfApi, folder: Path) -> str:
    if not folder.is_dir() or not any(folder.rglob("*.safetensors")):
        raise RuntimeError("No .safetensors adapter in checkpoint folder. Never upload placeholder files.")
    allowed = {".safetensors", ".json", ".md", ".yaml", ".yml", ".txt"}
    for entry in folder.rglob("*"):
        if entry.is_symlink():
            raise RuntimeError("Symlinks not allowed in checkpoint upload")
        if entry.is_file() and entry.suffix.lower() not in allowed:
            raise RuntimeError(f"Unsupported checkpoint file (privacy safeguard): {entry}")
    name = ensure_private_repo(api, "model")
    api.upload_folder(repo_id=name, repo_type="model", folder_path=str(folder),
                      commit_message="Publish reviewed private Latent Diaries reader checkpoint")
    return f"https://huggingface.co/{name}"


def publish_dataset(api: HfApi, folder: Path) -> str:
    # Dataset publishing intentionally separate: potentially identifying images/voices.
    if os.getenv("LD_APPROVE_DATASET_UPLOAD", "no") != "yes":
        raise RuntimeError("Require LD_APPROVE_DATASET_UPLOAD=yes for private dataset upload")
    if not folder.is_dir() or not any(folder.iterdir()):
        raise RuntimeError("No dataset files")
    name = ensure_private_repo(api, "dataset")
    api.upload_folder(repo_id=name, repo_type="dataset", folder_path=str(folder),
                      commit_message="Private training dataset snapshot")
    return f"https://huggingface.co/datasets/{name}"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["create", "publish", "dataset"])
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--folder", type=Path)
    args = p.parse_args()
    if not os.getenv("HF_TOKEN"):
        p.error("HF_TOKEN missing in .env")
    api = HfApi(token=os.getenv("HF_TOKEN"))
    if args.action == "create":
        print("Private HF repositories:", ensure_private_repo(api, "model"),
              ensure_private_repo(api, "dataset"))
    elif args.action == "publish":
        if not args.checkpoint:
            p.error("--checkpoint required")
        print(publish_model(api, args.checkpoint))
    else:
        if not args.folder:
            p.error("--folder required")
        print(publish_dataset(api, args.folder))

if __name__ == "__main__":
    main()
