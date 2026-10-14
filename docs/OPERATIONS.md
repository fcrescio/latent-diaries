# Operations: local Docker, GitHub, HF Hub, HF GPU Jobs

## Local prerequisites

Git, Python >=3.11, Docker Engine/Desktop + Compose v2, `gh` CLI for repository creation. For local NVIDIA GPU: driver and NVIDIA Container Toolkit; `docker run --gpus all ... nvidia-smi` should work. **Apple Silicon:** Docker's Linux backend normally does not have MPS GPU; plan to use HF Jobs or explicitly implement a native MPS runner as a documented exception if permitted.

```
cp .env.example .env
chmod 600 .env
# fill tokens/names and choose a strong REVIEW_PASSWORD
uv sync --locked --extra test
make test
make init
make demo  # SYNTHETIC image + sine audio: tests ONLY
make review
# http://localhost:8080
```

The database and media live under `run-data/`, intentionally not tracked. To ingest actual generated samples: `latent-diaries ingest-review --manifest run-data/manifests/your_run.jsonl`. Any real personal gallery should not be published to a Space without a separate security review.

### Docker worker

```
make test-docker
make gpu-doctor    # only if CUDA runtime is configured
# For M0: set LD_REF_AUDIO=/workspace/run-data/voices/reference.wav in .env
make m0            # CPU possible but slow; uses real Qwen weights
```

Downloads occur from HF into named cache volume. `qwen-baseline` does not run unless reference WAV exists (no voice data is distributed). Worker dependencies can be heavy. Record Docker image digest and package versions for each experiment.

The locked M0/M1 stack uses qwen-tts 0.1.1, Transformers 4.57.3,
Accelerate 1.12.0 and Hub 0.36.2. `make test-docker` runs the full regression
suite in the same worker, including tiny real-Qwen gradient/cache tests. Those
random-weight tests demonstrate tensor mechanics only, never audible Italian.
M0 writes `artifacts/m0_italian.json` and a reference codec-round-trip WAV;
human listening is still required. Set `LD_CODE_REVISION` to the source commit
before an experiment. Upstream model components are pinned via a full snapshot.

For this WSL checkout, native Docker/Compose/Buildx were installed under ignored
`.cache/` and use the existing Docker Desktop Unix socket. Reproduce checks:

```bash
DOCKER_CONFIG="$PWD/.cache/docker-config" make DOCKER=.cache/tools/docker test-docker
DOCKER_CONFIG="$PWD/.cache/docker-config" .cache/tools/docker compose --profile gpu run --rm experiment-gpu python -m pytest -q
```

After supplying a consented `run-data/voices/reference.wav` and setting
`LD_REF_AUDIO=/workspace/run-data/voices/reference.wav` in `.env`, launch M0:

```bash
LD_CODE_REVISION=$(git rev-parse HEAD) DOCKER_CONFIG="$PWD/.cache/docker-config" make DOCKER=.cache/tools/docker m0-gpu
```

Set `HF_NAMESPACE=fcrescio` and a strong `REVIEW_PASSWORD` locally before Hub
commands and listening review respectively. Keep secrets out of chat.

## Create GitHub repository

`make github` invokes `scripts/bootstrap_github.py` using `GH_TOKEN` from `.env`, initializes a git repo, checks for staged `.env`/media, commits and pushes to `GITHUB_OWNER/GITHUB_REPO`. Requires `gh` CLI and a token with suitable repo permissions. Default GitHub visibility private. Review before pushing: `git status --ignored`.

CI on pushes tests server/import/security and builds review image without a GPU. A **separate manual** `workflow_dispatch` builds/pushes worker image to GHCR. It does not train or incur cloud GPU cost.

## HF Hub checkpoints and data

```
make hf-create              # creates private model+dataset repos
# After trainer produces real safetensors+metadata at artifacts/checkpoint-ready:
make hf-publish             # uploads private adapter only
# Only after explicit dataset permission; include only consented data:
LD_APPROVE_DATASET_UPLOAD=yes make hf-dataset-publish
```

Always pin upstream model revision: `Qwen/Qwen3-TTS-12Hz-0.6B-Base` and `google/embeddinggemma-2`, original licenses/consent, training seed and git revision in checkpoint card. Don't redistribute upstream base weights.

## Remote training with HF Jobs

HF Jobs directly supports Docker image execution, GPUs, finite timeout, `HF_TOKEN` secrets and private Hub dataset mounts. Before use:

1. Implement and integration-test `train-reader`; set capability flag **only after proof**.
2. Create private HF dataset repository and upload permitted assets and manifest.
3. Build a **publicly pullable, secrets-free** worker container image to GHCR. Source GitHub repository may be private; confirm image visibility independently (GHCR packages can be private by default). Set `HF_JOB_IMAGE=ghcr.io/<owner>/<repo>-worker:<sha>`.
4. `hf jobs hardware` to check hardware flavors, current prices and quotas.
5. `make hf-job-plan` to inspect exact launch command with no tokens printed.
6. Human approves **one** paid job, sets `LD_APPROVE_PAID_JOBS=yes` in local `.env`, then `make hf-job-run`.
7. Immediately reset flag to `no`, record Job ID, monitor with `hf jobs ps`/`hf jobs logs`, cancel if needed. Implement real trainer to write checkpoints to private HF Hub during and at end of job; ephemeral files are not durable by themselves.

If the selected HF Job cannot read GHCR image due to image authentication, don't transmit registry secrets in command line; publish a sanitized public package or use HF-supported Docker Space images. Job script is fail-closed while trainer is unimplemented.

### Secrets in Jobs

`--secrets HF_TOKEN` passes the token through encrypted job secret. The shell command prints only the **name** of the secret, never its value. Other environment values (model IDs, namespace) can be passed with `-e`. Never use `--env-file .env` for cloud jobs because it could copy GH_TOKEN and REVIEW_PASSWORD.

## Reproducibility

Track git SHA, Docker image digest, HF repo revisions, seed, precision, max new audio tokens, GPU type, elapsed time, train/validation split by **walk / scene_group** not neighboring frames. Store run manifests, metrics, training logs and private checkpoints. Create an experiment manifest for each run. Without actual GPU execution, those numbers are unknown and must not be invented.
