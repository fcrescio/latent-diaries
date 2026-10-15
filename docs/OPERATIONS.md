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

`LD_REF_AUDIO` is the path inside the container: for host file
`run-data/voices/test.wav`, use `/workspace/run-data/voices/test.wav`.
`LD_REF_TEXT` contains that recording's exact transcript. Both stay in ignored
`.env`; the transcript is used only by the separate text-conditioned baseline.

After M0, run the pretrained assembly/cache controls against the cached snapshot:

```bash
DOCKER_CONFIG="$PWD/.cache/docker-config" .cache/tools/docker compose --profile gpu run --rm -v "$PWD/scripts:/workspace/scripts:ro" experiment-gpu python scripts/validate_qwen_pretrained.py --dtype float32
```

Results go to `artifacts/qwen-pretrained-validation.json`. A BF16 run currently
exceeds the cache comparison tolerance; FP32 is the verified mechanical control.
These controls use projected teacher text to compare stock assembly, and do not
constitute learned textless narration or image conditioning.

### M1 single-utterance inversion control

This trains only 16 continuous tokens against the M0 audio, with Qwen frozen.
The verified first experiment uses FP32 and explicit same-position loss for the
15 residual codebooks. Its waveform is an utterance memorization control, not
image-conditioned narration or generalization evidence.

```bash
LD_CODE_REVISION=$(git rev-parse HEAD) DOCKER_CONFIG="$PWD/.cache/docker-config" .cache/tools/docker compose --profile gpu run --rm experiment-gpu latent-diaries train-soft-prefix --steps 200 --run-dir artifacts/m1-new-run --residual-loss-weight 1
# Increase total steps while preserving the exact source/lock/config:
LD_CODE_REVISION=$(git rev-parse HEAD) DOCKER_CONFIG="$PWD/.cache/docker-config" .cache/tools/docker compose --profile gpu run --rm experiment-gpu latent-diaries train-soft-prefix --steps 400 --run-dir artifacts/m1-new-run --residual-loss-weight 1 --resume
# Fresh-worker inference accepts a checkpoint, speaker reference and duration.
DOCKER_CONFIG="$PWD/.cache/docker-config" .cache/tools/docker compose --profile gpu run --rm -e LD_REF_TEXT= experiment-gpu latent-diaries infer-soft-prefix --run-dir artifacts/m1-new-run --output artifacts/m1-new-run/reloaded.wav --max-frames 92 --seed 42
```

Before rebuilding the worker during development, bind the current source with
`-v "$PWD/src:/workspace/src:ro"` before the service name. Checkpoints contain
prefix safetensors, AdamW/RNG state, config and hashes; they reject mismatched
source/configuration and inconsistent prefix/state pairs. Gold codec targets
and logs remain local. Transcript-free inference does not read the target audio
or `LD_REF_TEXT`. The reloaded run-002 waveform exactly matches its original
generated waveform. No inference text is taken from checkpoint metadata.

The codec-0-only run-001 is a documented negative control: teacher-forced CE
became tiny but autoregressive output collapsed to 13 frames. Do not equate
training CE with intelligible speech. Run-002 added residual loss after diagnosis
and produced 50 frames; the user rejected both sampled and fully greedy speech
as meaningless. Details are in DECISIONS.

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

## Matched-teacher and autoregressive diagnostics

All commands below run locally in the same GPU worker. Rebuild with
`make test-docker` first. Prefixes/cached logits/audio stay in ignored artifacts.
The teacher uses training-only text; its projected prefix is a positive control,
not a learned or image-conditioned result. Inference loads only student tensors,
fixed protocol markers and the speaker reference.

```bash
DOCKER_CONFIG="$PWD/.cache/docker-config" .cache/tools/docker compose --profile gpu run --rm -e LD_REF_TEXT= experiment-gpu python scripts/cache_inversion_teacher.py --output-dir artifacts/m1-teacher-new
LD_CODE_REVISION=$(git rev-parse HEAD) DOCKER_CONFIG="$PWD/.cache/docker-config" .cache/tools/docker compose --profile gpu run --rm -e LD_REF_TEXT= experiment-gpu latent-diaries train-soft-prefix --steps 200 --run-dir artifacts/m1-kl-new --target-audio artifacts/m1-teacher-new/teacher.wav --residual-loss-weight 1 --teacher-cache-dir artifacts/m1-teacher-new
# Matched-target CE ablation: same command, fresh directory, add --cached-gold-ce.
# Capacity ablation: same KL command, fresh directory, add --tokens 32.
# Generated-history phase: fresh directory, add --init-prefix-dir artifacts/m1-kl-new --rollout-every 50 --learning-rate 0.01.
# Resume that phase with identical flags plus --resume and increased total --steps.
DOCKER_CONFIG="$PWD/.cache/docker-config" .cache/tools/docker compose --profile gpu run --rm -e LD_REF_TEXT= experiment-gpu python scripts/probe_policy_gap.py --run-dir artifacts/m1-kl-new --teacher-cache-dir artifacts/m1-teacher-new
DOCKER_CONFIG="$PWD/.cache/docker-config" .cache/tools/docker compose --profile gpu run --rm experiment-gpu python scripts/asr_diagnostic.py artifacts/m1-kl-new/trained.wav --output artifacts/asr-new.json --revision 169d4a4341b33bc18d8881c4b69c2e104e1cc0af
```

Generated-history supervision saves immutable rollout caches and checks their
hash on resume; the 50→100 step continuation was exercised. New-phase
initialization copies prefix tensors and resets the optimizer. Student and teacher
can have different prefix lengths; codec-0 alignment includes EOS, residual
alignment uses same-frame heads. KL is diagnostic, never a human quality score.
Local Whisper is auxiliary only, uploads no audio, and never feeds a transcript
back into synthesis. None of these ablations has established M1 acceptance.


To repeat decoding controls on the unchanged 32-token candidate:

```bash
DOCKER_CONFIG="$PWD/.cache/docker-config" .cache/tools/docker compose --profile gpu run --rm -e LD_REF_TEXT= experiment-gpu python scripts/probe_autoregression.py --run-dir artifacts/m1-inversion-006
```

`probe_sampled.wav` is an exact replay of the original. `probe_codec0_greedy.wav`
removes first-codebook sampling; `probe_greedy.wav` removes sampling in all
codebooks. Oracle residual injection writes diagnostic numbers only, never audio.
The user accepts the codec-0-greedy / sampled-residual control as fully
intelligible Italian; the original sampled waveform remains only partly clear.


Reproduce the accepted single-utterance control through the regular inference CLI
(after rebuilding the worker). No teacher cache, gold audio codes or transcript
is read by this command:

```bash
DOCKER_CONFIG="$PWD/.cache/docker-config" .cache/tools/docker compose --profile gpu run --rm -e LD_REF_TEXT= experiment-gpu latent-diaries infer-soft-prefix --run-dir artifacts/m1-inversion-006 --output artifacts/m1-inversion-006/accepted-reloaded.wav --max-frames 99 --seed 42 --codec0-greedy
```

The flag disables sampling only for codec-0; all 15 residual codebooks retain
sampling. Omit it to preserve the earlier inference default. The output manifest
records this setting, seed and duration cap. Human acceptance is specific to the
accepted waveform/checkpoint/settings, not arbitrary seeds or novel utterances.


Distinct-utterance repetition (separate learned prefix; not unseen-utterance
generalization):

```bash
DOCKER_CONFIG="$PWD/.cache/docker-config" .cache/tools/docker compose --profile gpu run --rm -e LD_REF_TEXT= experiment-gpu python scripts/cache_inversion_teacher.py --output-dir artifacts/m1-teacher-002 --text "Sotto la pioggia, una finestra illuminata sembrava custodire un segreto."
LD_CODE_REVISION=$(git rev-parse HEAD) DOCKER_CONFIG="$PWD/.cache/docker-config" .cache/tools/docker compose --profile gpu run --rm -e LD_REF_TEXT= experiment-gpu latent-diaries train-soft-prefix --steps 200 --tokens 32 --run-dir artifacts/m1-inversion-007 --target-audio artifacts/m1-teacher-002/teacher.wav --residual-loss-weight 1 --teacher-cache-dir artifacts/m1-teacher-002
DOCKER_CONFIG="$PWD/.cache/docker-config" .cache/tools/docker compose --profile gpu run --rm -e LD_REF_TEXT= experiment-gpu latent-diaries infer-soft-prefix --run-dir artifacts/m1-inversion-007 --output artifacts/m1-inversion-007/codec0-greedy.wav --max-frames 105 --seed 42 --codec0-greedy
```

Use fresh directories when repeating training/teacher caching. Sentence text is
used only for teacher supervision; the final command reads no transcript.


## Shared numeric-factor probe (research, no images)

The design `configs/research/compositions-v1.json` fixes six training combinations
and two held-out combinations before any optimization. Its three numeric factors
are authored semantic inputs. Teacher strings are used only to cache supervision;
this experiment does not use text embeddings as inference inputs and does not
establish image grounding. All artifacts stay local.

```bash
DOCKER_CONFIG="$PWD/.cache/docker-config" .cache/tools/docker compose --profile gpu run --rm -e LD_REF_TEXT= experiment-gpu python scripts/prepare_shared_probe.py --design configs/research/compositions-v1.json --output-dir artifacts/shared-corpus-001
LD_CODE_REVISION=$(git rev-parse HEAD) DOCKER_CONFIG="$PWD/.cache/docker-config" .cache/tools/docker compose --profile gpu run --rm -e LD_REF_TEXT= experiment-gpu python scripts/shared_probe.py train --dataset artifacts/shared-corpus-001/dataset.json --run-dir artifacts/shared-mapper-001 --steps 600
LD_CODE_REVISION=$(git rev-parse HEAD) DOCKER_CONFIG="$PWD/.cache/docker-config" .cache/tools/docker compose --profile gpu run --rm -e LD_REF_TEXT= experiment-gpu python scripts/shared_probe.py train --dataset artifacts/shared-corpus-001/dataset.json --run-dir artifacts/shared-mapper-001 --steps 1200 --resume
DOCKER_CONFIG="$PWD/.cache/docker-config" .cache/tools/docker compose --profile gpu run --rm -e LD_REF_TEXT= experiment-gpu python scripts/shared_probe.py evaluate --inputs artifacts/shared-corpus-001/eval-inputs.json --run-dir artifacts/shared-mapper-001 --output-dir artifacts/shared-eval-001 --ids c01 c00 c07
```

Rebuild the worker before these commands, or bind current `src` and `scripts`
read-only before the service name. Fresh directories prevent overwriting previous
runs. Preparation can reuse complete teacher caches after verifying their text
fingerprints. Training opens only train-split teacher tensors, freezes the Qwen
backbone and logs per-example KL. Checkpoint resume validates source/config/weights
and restores AdamW; example order is deterministic and contains no dropout or
sampling. Evaluation opens only shared mapper tensors, numeric inputs and frozen
backbone/speaker; its spec rejects text/teacher paths. The zero-input waveform is
reused across cases, so repeated null entries are not independent samples.

The evaluator compares both decoding modes at seed 42 with the same 105-frame
cap. Results need human listening; low training KL, duration and auxiliary ASR
cannot establish intelligibility or generalization.


Post-training diagnostic and private audio-only listening page:

```bash
DOCKER_CONFIG="$PWD/.cache/docker-config" .cache/tools/docker compose --profile gpu run --rm -e LD_REF_TEXT= -v "$PWD/scripts:/workspace/scripts:ro" experiment-gpu python scripts/diagnose_shared_probe.py --dataset artifacts/shared-corpus-001/dataset.json --run-dir artifacts/shared-mapper-001 --output artifacts/shared-mapper-001/teacher-forced-diagnostic.json
.venv/bin/python scripts/blind_audio_probe.py --results artifacts/shared-eval-001/results.json --output-dir artifacts/shared-listening-001
```

The diagnostic may read held-out teacher logits only after freezing the completed
checkpoint; it does not update weights or select a checkpoint. The standalone
evaluation command above reads no teacher logits. The listening page shuffles
and renames eight unique WAVs, stores a private condition mapping, and does not
fabricate ratings. It is a local file, with no HTTP listener or public exposure.
This audio-only probe cannot supply image-grounding/art-gallery evidence.


## Additive development control

After blinded review of shared-mapper-001, c00/c07 are exposed development cases.
Keep them out of optimization, but do not call them a fresh untouched test set
when comparing later architectures. The additive map (`width=0`) has a shared
bias plus three feature directions, 131,072 trainable parameters and no
cross-factor interactions. The existing MLP remains the default; old MLP
checkpoint layouts still load for inference. Resume requires exact source as
before.

```bash
LD_CODE_REVISION=$(git rev-parse HEAD) DOCKER_CONFIG="$PWD/.cache/docker-config" .cache/tools/docker compose --profile gpu run --rm -e LD_REF_TEXT= experiment-gpu python scripts/shared_probe.py train --dataset artifacts/shared-corpus-001/dataset.json --run-dir artifacts/shared-mapper-002-additive --steps 1200 --architecture additive --learning-rate 0.03
DOCKER_CONFIG="$PWD/.cache/docker-config" .cache/tools/docker compose --profile gpu run --rm -e LD_REF_TEXT= experiment-gpu python scripts/diagnose_shared_probe.py --dataset artifacts/shared-corpus-001/dataset.json --run-dir artifacts/shared-mapper-002-additive --output artifacts/shared-mapper-002-additive/teacher-forced-diagnostic.json
```

The larger learning rate follows the direct-prefix parameterization used in the
single-utterance controls. This comparison changes architecture and optimization
scale; it does not isolate architecture as the sole causal factor. Gold-history
KL remains a technical diagnostic, not human intelligibility.


## Direct teacher-prefix supervision control

This changes supervision from audio-logit KL to continuous teacher-prefix vectors
and prefix length. Fit an affine numeric-factor map using only six training
caches, right-padding to their maximum length; predict/trim the prefix length
from numeric features during evaluation. This is a development diagnostic, not
an image reader or a fresh held-out test. No teacher files/text are read by the
evaluation command.

```bash
DOCKER_CONFIG="$PWD/.cache/docker-config" .cache/tools/docker compose --profile experiment run --rm experiment python scripts/inspect_factor_geometry.py --dataset artifacts/shared-corpus-001/dataset.json --output artifacts/shared-factor-geometry.json
LD_CODE_REVISION=$(git rev-parse HEAD) DOCKER_CONFIG="$PWD/.cache/docker-config" .cache/tools/docker compose --profile experiment run --rm experiment python scripts/prefix_regression_probe.py fit --dataset artifacts/shared-corpus-001/dataset.json --run-dir artifacts/shared-prefix-regression-001
DOCKER_CONFIG="$PWD/.cache/docker-config" .cache/tools/docker compose --profile gpu run --rm -e LD_REF_TEXT= experiment-gpu python scripts/prefix_regression_probe.py evaluate --inputs artifacts/shared-corpus-001/eval-inputs.json --run-dir artifacts/shared-prefix-regression-001 --output-dir artifacts/shared-prefix-regression-eval-001 --ids c00 c01 c07
.venv/bin/python scripts/blind_audio_probe.py --results artifacts/shared-prefix-regression-eval-001/results.json --output-dir artifacts/prefix-regression-listening-002 --first-label P
```

Rebuild the one worker first or bind current `src` and `scripts` read-only before
the service name. Use fresh fit/listening directories. Closed-form fit needs no
GPU or optimizer continuation. Evaluation uses fixed seed 42, sampled codec-0
and residuals, a 105-frame cap and a separate null factor input. Tensor length
prediction and trimming are tested; human intelligibility remains a listening
gate. A new untouched test set is required before claiming generalization.


## Frozen fresh composition test (v2)

The v1 development cases c00/c07 were exposed while choosing prefix supervision.
The v2 design excludes them and reserves s01/s02 before new generation or fit.
Use three weather one-hot numeric coordinates plus opening/memory coordinates.
Eight training rows include the original six and two snow cases; the two fresh
snow combinations contain factor levels already represented in training. Keep
the same affine prefix/length method, sampled decoder, seed and duration cap.

```bash
DOCKER_CONFIG="$PWD/.cache/docker-config" .cache/tools/docker compose --profile gpu run --rm -e LD_REF_TEXT= experiment-gpu python scripts/prepare_shared_probe.py --design configs/research/compositions-v2.json --output-dir artifacts/shared-corpus-002 --reuse-cache-root artifacts/shared-corpus-001
LD_CODE_REVISION=$(git rev-parse HEAD) DOCKER_CONFIG="$PWD/.cache/docker-config" .cache/tools/docker compose --profile experiment run --rm experiment python scripts/prefix_regression_probe.py fit --dataset artifacts/shared-corpus-002/dataset.json --run-dir artifacts/shared-prefix-regression-002 --evaluation-scope fresh-test
DOCKER_CONFIG="$PWD/.cache/docker-config" .cache/tools/docker compose --profile gpu run --rm -e LD_REF_TEXT= experiment-gpu python scripts/prefix_regression_probe.py evaluate --inputs artifacts/shared-corpus-002/eval-inputs.json --run-dir artifacts/shared-prefix-regression-002 --output-dir artifacts/shared-prefix-regression-eval-002 --ids s00 s01 s02
.venv/bin/python scripts/blind_audio_probe.py --results artifacts/shared-prefix-regression-eval-002/results.json --output-dir artifacts/prefix-regression-listening-003 --first-label T
```

Reuse checks sentence fingerprints and writes references to old caches rather
than duplicating media. Fit opens train rows only. Fresh-test labels identify
mapper-level combinatorial holdouts, not new vocabulary or unseen images.
Human listening and condition sensitivity are still needed before accepting M1.2.


Read-only post-test diagnosis (after human feedback; no weight updates):

```bash
DOCKER_CONFIG="$PWD/.cache/docker-config" .cache/tools/docker compose --profile experiment run --rm -v "$PWD/scripts:/workspace/scripts:ro" experiment python scripts/diagnose_prefix_regression.py --dataset artifacts/shared-corpus-002/dataset.json --run-dir artifacts/shared-prefix-regression-002 --output artifacts/prefix-regression-002-posttest-diagnostic.json
```

This compares predicted/teacher continuous vectors and lengths. First-five versus
remaining-position errors are tensor diagnostics, not word-aligned speech scores.
The initial fresh test is now exposed after listening; do not reuse its cases as
untouched evaluation when changing architecture or supervision.
