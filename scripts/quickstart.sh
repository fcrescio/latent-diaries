#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
[[ -f .env ]] || cp .env.example .env
chmod 600 .env
mkdir -p run-data/review-assets artifacts
python -m pip install -e '.[test]'
python -m pytest -q
printf '\nBootstrap ready. Fill .env, run: make demo && make review\n'
