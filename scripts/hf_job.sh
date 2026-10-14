#!/usr/bin/env bash
# Launch a Docker image as an HF Job; never charges without the explicit paid-job gate.
set -euo pipefail
cd "$(dirname "$0")/.."
set -a
# shellcheck disable=SC1091
source .env
set +a
MODE="${1:---dry-run}"
if [[ "$MODE" != "--dry-run" && "$MODE" != "--execute" ]]; then
  echo "Use --dry-run or --execute" >&2; exit 2
fi
: "${HF_JOB_IMAGE:?Publish a public, secrets-free worker image and set HF_JOB_IMAGE}"
: "${HF_NAMESPACE:?Set HF_NAMESPACE}"
: "${HF_DATASET_REPO:?Set HF_DATASET_REPO}"
: "${HF_MODEL_REPO:?Set HF_MODEL_REPO}"
DATASET="${HF_DATASET_REPO}"
[[ "$DATASET" == */* ]] || DATASET="$HF_NAMESPACE/$DATASET"
MODEL="${HF_MODEL_REPO}"
[[ "$MODEL" == */* ]] || MODEL="$HF_NAMESPACE/$MODEL"
# Cloud image should include working trainer; otherwise don't even allow expensive execution.
if [[ "$MODE" == "--execute" ]]; then
  python - <<'PY'
import json
c=json.load(open('docs/capabilities.json'))
assert c['image_reader_training_implemented'], 'Implement/test image reader training first; refusing paid job'
PY
  [[ "${LD_APPROVE_PAID_JOBS:-no}" == "yes" ]] || { echo "Paid job not approved. Set LD_APPROVE_PAID_JOBS=yes in .env." >&2; exit 3; }
  : "${HF_TOKEN:?HF_TOKEN needed}"
fi
CMD=(hf jobs run --flavor "${HF_JOB_FLAVOR:-a10g-small}" --timeout "${HF_JOB_TIMEOUT:-2h}"
  --secrets HF_TOKEN
  -e "HF_MODEL_REPO=$MODEL"
  -e "HF_NAMESPACE=$HF_NAMESPACE"
  -e "LD_VISION_MODEL=${LD_VISION_MODEL:-google/embeddinggemma-2}"
  -e "LD_QWEN_MODEL=${LD_QWEN_MODEL:-Qwen/Qwen3-TTS-12Hz-0.6B-Base}"
  -v "hf://datasets/$DATASET:/dataset:ro"
  "$HF_JOB_IMAGE" -- latent-diaries train-reader)
if [[ "$MODE" == "--dry-run" ]]; then
  printf 'DRY RUN (no charge): '; printf '%q ' "${CMD[@]}"; printf '\n'
  echo 'Verify with: hf jobs hardware ; confirm image public and dataset private.'
else
  "${CMD[@]}"
fi
