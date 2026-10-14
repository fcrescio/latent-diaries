"""Create a GitHub repo and push using GH_TOKEN; no credentials embedded in remotes."""
from __future__ import annotations
import os
import subprocess
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def run(*args: str, env: dict[str, str] | None = None):
    return subprocess.run(args, check=True, env=env)


def main():
    token, owner = os.getenv("GH_TOKEN"), os.getenv("GITHUB_OWNER")
    repo = os.getenv("GITHUB_REPO", "latent-diaries")
    if not token or not owner:
        raise SystemExit("Set GH_TOKEN and GITHUB_OWNER in .env")
    if not Path(".env").exists() or not Path(".gitignore").exists():
        raise SystemExit("Run in project root")
    env = dict(os.environ, GH_TOKEN=token, GITHUB_TOKEN=token)
    run("gh", "--version", env=env)
    if not Path(".git").exists():
        run("git", "init", "-b", "main")
    run("git", "add", ".")
    tracked = subprocess.check_output(["git", "ls-files"], text=True).splitlines()
    if any(p == ".env" or p.startswith("run-data/") or p.startswith("artifacts/") for p in tracked):
        raise SystemExit("Refusing to push: secrets/data path staged in git")
    if subprocess.run(["git", "diff", "--cached", "--quiet"]).returncode != 0:
        run("git", "commit", "-m", "Bootstrap Latent Diaries")
    visibility = "--private" if os.getenv("GITHUB_PRIVATE", "true").lower() == "true" else "--public"
    target = f"{owner}/{repo}"
    # Existing repos are supported for interrupted/restarted bootstraps.
    exists = subprocess.run(["gh", "repo", "view", target], env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    if exists:
        desired = f"https://github.com/{target}.git"
        origin = subprocess.run(["git", "remote", "get-url", "origin"], capture_output=True, text=True)
        if origin.returncode != 0:
            run("git", "remote", "add", "origin", desired)
        elif origin.stdout.strip().rstrip("/") not in {desired, desired.removesuffix(".git")}:
            raise SystemExit("Existing 'origin' points to a different repository; resolve manually")
        run("gh", "auth", "setup-git", env=env)
        run("git", "push", "-u", "origin", "HEAD:main", env=env)
    else:
        # Requires gh CLI authenticated via GH_TOKEN. Create is explicit, never in make init.
        run("gh", "repo", "create", target, visibility, "--source", ".", "--remote", "origin", "--push", env=env)
    print("GitHub:", f"https://github.com/{target}")

if __name__ == "__main__":
    main()
