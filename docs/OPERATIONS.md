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


## Sequence-aware reader v3 versus affine baseline

`configs/research/compositions-v3.json` was committed before teacher generation:
22 training combinations, fresh w02/e01 tests, six weather categories. Previously
exposed cases are now training examples. Teacher-prefix reconstruction consumes
numeric factors only. It is not an image reader or unseen-vocabulary evaluation.

Rebuild the single worker first (or bind current `src` and `scripts` read-only
before the service name during development). Use fresh fit/listening directories.

```bash
DOCKER_CONFIG="$PWD/.cache/docker-config" .cache/tools/docker compose --profile gpu run --rm -e LD_REF_TEXT= experiment-gpu python scripts/prepare_shared_probe.py --design configs/research/compositions-v3.json --output-dir artifacts/shared-corpus-003 --reuse-cache-root artifacts/shared-corpus-001 --reuse-cache-root artifacts/shared-corpus-002
LD_CODE_REVISION=$(git rev-parse HEAD) DOCKER_CONFIG="$PWD/.cache/docker-config" .cache/tools/docker compose --profile gpu run --rm experiment-gpu python scripts/sequence_probe.py train --dataset artifacts/shared-corpus-003/dataset.json --run-dir artifacts/sequence-reader-001 --steps 800 --schedule-steps 1600
LD_CODE_REVISION=$(git rev-parse HEAD) DOCKER_CONFIG="$PWD/.cache/docker-config" .cache/tools/docker compose --profile gpu run --rm experiment-gpu python scripts/sequence_probe.py train --dataset artifacts/shared-corpus-003/dataset.json --run-dir artifacts/sequence-reader-001 --steps 1600 --schedule-steps 1600 --resume
LD_CODE_REVISION=$(git rev-parse HEAD) DOCKER_CONFIG="$PWD/.cache/docker-config" .cache/tools/docker compose --profile experiment run --rm experiment python scripts/prefix_regression_probe.py fit --dataset artifacts/shared-corpus-003/dataset.json --run-dir artifacts/affine-baseline-v3 --evaluation-scope fresh-test
DOCKER_CONFIG="$PWD/.cache/docker-config" .cache/tools/docker compose --profile gpu run --rm -e LD_REF_TEXT= experiment-gpu python scripts/sequence_probe.py evaluate --inputs artifacts/shared-corpus-003/eval-inputs.json --run-dir artifacts/sequence-reader-001 --output-dir artifacts/sequence-reader-eval-001 --ids w00 w02 e01
DOCKER_CONFIG="$PWD/.cache/docker-config" .cache/tools/docker compose --profile gpu run --rm -e LD_REF_TEXT= experiment-gpu python scripts/prefix_regression_probe.py evaluate --inputs artifacts/shared-corpus-003/eval-inputs.json --run-dir artifacts/affine-baseline-v3 --output-dir artifacts/affine-baseline-v3-eval --ids w00 w02 e01
DOCKER_CONFIG="$PWD/.cache/docker-config" .cache/tools/docker compose --profile experiment run --rm experiment python scripts/diagnose_sequence_probe.py --dataset artifacts/shared-corpus-003/dataset.json --run-dir artifacts/sequence-reader-001 --output artifacts/sequence-reader-001/prefix-diagnostic.json
DOCKER_CONFIG="$PWD/.cache/docker-config" .cache/tools/docker compose --profile experiment run --rm experiment python scripts/diagnose_prefix_regression.py --dataset artifacts/shared-corpus-003/dataset.json --run-dir artifacts/affine-baseline-v3 --output artifacts/affine-baseline-v3/prefix-diagnostic.json
```

The 1,600-step schedule is fixed during the first 800-step phase, so resume does
not restart decay. Checkpoints verify model/config/source/prefix fingerprints and
restore optimizer plus RNG. Normalization and length prediction use train rows
only; no held-out prefix is opened until frozen post-training diagnostics.
Generation reads no teacher/text/code targets. Training loss is normalized masked
MSE; post-training comparisons use valid-position raw MSE, not human quality.

Combine private result manifests and generate new blinded labels without serving
files publicly (this host-side operation is JSON/file preparation, not training):

```bash
.venv/bin/python - <<'PYMERGE'
import json
from pathlib import Path
folder = Path("artifacts/sequence-comparison-001")
folder.mkdir(exist_ok=True)
rows = []
for method, source in [("sequence", "sequence-reader-eval-001"), ("affine", "affine-baseline-v3-eval")]:
    values = json.loads(Path(f"artifacts/{source}/results.json").read_text())
    rows.extend(row | {"method": method} for row in values["results"])
(folder / "results.json").write_text(json.dumps({"results": rows}, indent=2) + "\n")
PYMERGE
.venv/bin/python scripts/blind_audio_probe.py --results artifacts/sequence-comparison-001/results.json --output-dir artifacts/sequence-listening-001 --label-prefix E3- --shuffle-seed 2026101503
```

Private mappings are 0600; no ratings are synthesized. Both methods use seed 42,
sampled primary/residual codebooks and cap105. Compare fresh cases and controls
through human listening before updating generalization capability flags.


2026-10-15 listening is complete: the frozen sequence checkpoint passes both
prespecified fresh combinations and the training control; null is rejected.
This accepts scoped M1.2 for numeric factors at seed 42. The affine comparison
does not pass fresh intelligibility. Preserve this checkpoint and split; all
reviewed test cases are now exposed. Image grounding and seed robustness remain
unverified. Sanitized evidence is in `validation/2026-10-15-sequence-reader.json`;
individual judgments and audio fingerprints remain in ignored local artifacts.


## M2 frozen image-feature cache (implemented and exercised)

Build the single worker and prepare two attributed NASA images locally. The
preparation script verifies fixed file hashes; this pair tests encoder mechanics,
not a grounding benchmark or training corpus. NASA credits and its image-usage
policy URL are retained in the local manifest. No media is uploaded.

```bash
DOCKER_CONFIG="$PWD/.cache/docker-config" .cache/tools/docker compose --profile experiment build experiment
.venv/bin/python scripts/prepare_vision_smoke.py
LD_CODE_REVISION=$(git rev-parse HEAD) DOCKER_CONFIG="$PWD/.cache/docker-config" .cache/tools/docker compose --profile gpu run --rm experiment-gpu /opt/vision/.venv/bin/python scripts/cache_image_features.py --manifest artifacts/m2-image-smoke-inputs.json --output-dir artifacts/m2-vision-fresh --revision 914f7f89142e33e77833254d9c9b90c3cef7303b --device cuda
```

Use a fresh output directory. For CPU, use the `experiment` service and
`--device cpu`; the full pretrained CPU path has not yet been exercised.
`make test-docker` includes the small official vision API/processor tests on CPU.
GPU extraction is verified at `artifacts/m2-vision-smoke-004/manifest.json`;
sanitized evidence is in `validation/2026-10-15-image-features.json`.

Custom manifests are nonempty JSON lists of objects with exactly `id`, `image`,
`scene_group`, `source`, and `license`, all nonempty strings. Image paths are local
paths relative to the worker's `/workspace`; provenance is metadata, never model
conditioning. Keep related frames in the same scene group when defining future
training/holdout splits. No captions or transcripts are accepted in this manifest.
Outputs are `[N,768]` normalized `features.safetensors` plus a manifest recording
model revision, both lock hashes, source hashes, source-image fingerprints,
processor shapes, precision, latency, frozen parameters, repeat/batch parity and
peak allocated VRAM. Source photos remain outside the cache; local smoke media
can be deleted independently by removing `run-data/images/m2-smoke` and its feature
cache. No automatic media retention/upload is enabled.

The next M2 experiment requires paired image/Italian teacher supervision, an
image reader trained only on training scenes, and a separate held-out-image
comparison against shuffled and null images at the same speaker/seed. The
accepted numeric M1 reader does not by itself meet this requirement.


## Private image-reader pilot (local only)

The user-authorized archive stays outside the repository and Docker build
context. Mount it read-only at `/private/photos` for feature extraction; subsequent
reader training and audio generation need cached vectors only. Never upload the
images, archive filenames/locations, labels, features, audio, weights, ratings or
per-image manifests. The local vision label service is an explicitly authorized
existing container; the label helper stops it in `finally`, including failures.
Do not start EmbeddingGemma/TTS training while that service occupies VRAM.

Generic preparation commands (supply local paths/container name yourself; keep
all outputs under ignored private storage):

```bash
.venv/bin/python scripts/inventory_private_images.py --root "$LD_PRIVATE_PHOTO_ROOT" --output artifacts/private-new/inventory.json
DOCKER_CONFIG="$PWD/.cache/docker-config" .venv/bin/python scripts/label_private_images.py --inventory artifacts/private-new/inventory.json --output artifacts/private-new/labels.jsonl --docker .cache/tools/docker --container "$LD_LABEL_CONTAINER"
.venv/bin/python scripts/select_private_pilot.py --inventory artifacts/private-new/inventory.json --labels artifacts/private-new/labels.jsonl --archive-root "$LD_PRIVATE_PHOTO_ROOT" --output-dir artifacts/private-new/split
```

Review the local candidate labels/exclusions and freeze the split before fitting.
`configs/research/image-pilot-v1.json` specifies the fixed recipe. Use
`cache_image_features.py` in the vision interpreter with `HF_HUB_OFFLINE=1` and
the read-only archive mount. Then run `prepare_image_prefixes.py` in the primary
GPU worker to project **training text only**, followed by `image_probe.py train`
for 800 updates and `--resume` to 1600 with `--schedule-steps 1600` both times.
`image_probe.py evaluate` accepts only the separate vector evaluation spec,
never the training design/labels. Use `LD_REF_TEXT=` and offline mode.

Exact commands for the exercised private run are saved locally in
`artifacts/private-photos-001/reproduce.md`; complete provenance/checkpoint/media
are deliberately absent from public Git/HF. All fits, optimizer steps and model
inference run in the managed worker; host helpers only prepare media/JSON and
call the authorized local labelling API. The inventorier was reproduced against
the frozen first four source rows; the full frozen inventory is retained.

`blind_private_gallery.py` creates resized EXIF-free review copies, assigns
matched/shuffled/null pairs, preserves the 0600 mapping, and imports real clips
without creating ratings. The current batch is E4-A…E4-M. Review remains behind
HTTP Basic on loopback. `REVIEW_PORT` changes only the host port (default 8080);
the exercised local instance uses 18080 because another service occupied 8080.
Visit `http://localhost:18080/#m2p1-a` with existing review credentials.
No public hosting or private-data upload is authorized. Review is complete for this batch: all four tested held-out outputs fail
intelligibility; only the training control is almost good. M2 remains incomplete.
Preserve these outputs and weights; reviewed cases are now exposed development
inputs and must not be reused as untouched tests for a revised recipe.


After the completed private review, run the frozen diagnostic with a fresh
output directory. This command intentionally opens held-out teacher text for
post-test error analysis, so these cases cannot become untouched future tests.
It never updates weights and its exact-text/oracle-length outputs are diagnostic
controls, not learned image-model results. All outputs must remain private.

```bash
DOCKER_CONFIG="$PWD/.cache/docker-config" .cache/tools/docker compose --profile gpu run --rm -e HF_HUB_OFFLINE=1 -e LD_REF_TEXT= -v "$PWD/src:/workspace/src:ro" -v "$PWD/scripts:/workspace/scripts:ro" experiment-gpu python scripts/diagnose_image_probe.py --design artifacts/private-photos-001/split-001/design.json --dataset artifacts/private-photos-001/prefixes-001/dataset.json --run-dir artifacts/private-photos-001/reader-001 --plan artifacts/private-photos-001/evaluation-plan.json --output-dir artifacts/private-photos-001/postreview-new
```

During development bind current scripts/src read-only before the service name;
the current already-built worker predates this diagnostic helper. Executed
private evidence is at `postreview-001/diagnostic.json` with source and unchanged
weight fingerprints. Auxiliary ASR transcripts remain private and are not human
quality scores. No VLM service restart or reader retraining was needed.


## Image-query/length development ablation

The exact-text positive has human acceptance. The following architecture control
uses only the already exposed 50/12 corpus, and cannot establish fresh-test
success. It uses the same single worker/dependency locks, with current source
and scripts bound read-only during development. No label-service restart needed.

```bash
LD_CODE_REVISION=$(git rev-parse HEAD) DOCKER_CONFIG="$PWD/.cache/docker-config" .cache/tools/docker compose --profile gpu run --rm -v "$PWD/src:/workspace/src:ro" -v "$PWD/scripts:/workspace/scripts:ro" experiment-gpu python scripts/image_probe.py train --dataset artifacts/private-photos-001/prefixes-001/dataset.json --run-dir artifacts/private-photos-001/query-reader-new --steps 800 --schedule-steps 1600 --architecture image-query
LD_CODE_REVISION=$(git rev-parse HEAD) DOCKER_CONFIG="$PWD/.cache/docker-config" .cache/tools/docker compose --profile gpu run --rm -v "$PWD/src:/workspace/src:ro" -v "$PWD/scripts:/workspace/scripts:ro" experiment-gpu python scripts/image_probe.py train --dataset artifacts/private-photos-001/prefixes-001/dataset.json --run-dir artifacts/private-photos-001/query-reader-new --steps 1600 --schedule-steps 1600 --architecture image-query --resume
DOCKER_CONFIG="$PWD/.cache/docker-config" .cache/tools/docker compose --profile gpu run --rm -e HF_HUB_OFFLINE=1 -e LD_REF_TEXT= -v "$PWD/src:/workspace/src:ro" -v "$PWD/scripts:/workspace/scripts:ro" experiment-gpu python scripts/diagnose_image_development.py --dataset artifacts/private-photos-001/prefixes-001/dataset.json --design artifacts/private-photos-001/split-001/design.json --baseline-dir artifacts/private-photos-001/reader-001 --query-dir artifacts/private-photos-001/query-reader-new --plan artifacts/private-photos-001/evaluation-plan.json --output-dir artifacts/private-photos-001/development-new
```

Exact completed private results are in `query-reader-001` and `development-v2`.
The v2 protocol was committed before fitting. OLS/ridge comparison changes only
length trimming, with train-only ridge alpha selection. Oracle target texts are
opened for frozen development diagnostics only. The audio gate (valid-position
MSE <1e-5) failed; no new listening batch or quality acceptance was created.
Never upload these per-image diagnostics, features or checkpoints. Original
checkpoint inference remains supported; use the recorded source revision when
resuming older runs rather than bypassing source-fingerprint protection.

## Canonical supervision / broader-data development control

`configs/research/canonical-image-v3.json` freezes this limited-repertoire
control. Reserve fresh scene groups before labelling. Pass only train/development
rows to `label_private_images.py --mode scene-v3`; the helper stops the local
vision container, including on failure. Choose authored equal-length sentences
in the CPU worker with `choose_canonical_texts.py --output <private-json>` before
seeing image labels. It uses the installed Qwen processor options; full-wrapper
projection later asserts identical lengths.

`build_canonical_design.py` takes `--inventory`, `--labels`, `--canonical-texts`,
`--archive-root` and a fresh `--output-dir`. It refuses fresh-test rows and
applies precommitted group/category caps. Extract only new selected images with
the usual read-only archive mount and vision worker. Then prepare cached
training targets (all paths below are placeholders for ignored local outputs):

```bash
worker=(.cache/tools/docker compose --profile gpu run --rm -e HF_HUB_OFFLINE=1 -e LD_REF_TEXT= -v "$PWD/src:/workspace/src:ro" -v "$PWD/scripts:/workspace/scripts:ro")
"${worker[@]}" experiment-gpu python scripts/prepare_canonical_prefixes.py --design artifacts/private-new/design/design.json --canonical-texts artifacts/private-new/canonical.json --features-dir artifacts/private-old/features --features-dir artifacts/private-new/features --output-dir artifacts/private-new/targets
"${worker[@]}" experiment-gpu python scripts/image_probe.py train --dataset artifacts/private-new/targets/dataset-expanded.json --run-dir artifacts/private-new/reader-expanded --steps 800 --schedule-steps 1600 --fixed-length --evaluation-scope development
"${worker[@]}" experiment-gpu python scripts/image_probe.py train --dataset artifacts/private-new/targets/dataset-expanded.json --run-dir artifacts/private-new/reader-expanded --steps 1600 --schedule-steps 1600 --fixed-length --evaluation-scope development --resume
```

Repeat the two fit commands with `dataset-old.json` and a new `reader-old`
directory for the original-cohort control. Keep source fixed across resume.
`diagnose_canonical_reader.py --targets-dir <targets> --old-run <reader-old>
--expanded-run <reader-expanded> --output <new-private-diagnostic.json>` runs
in the managed worker, opens cached supervision only for frozen development
analysis, checks both weight hashes and never generates audio or updates weights.
Never pass this training dataset to standalone inference. Final research
inference uses the separate strict numeric `eval-*.json`; category/text targets
are absent. `--allow-new-images` explicitly permits new IDs without editing a
frozen checkpoint, but does not waive the experimental fresh-access gate.

The completed local run uses `artifacts/private-photos-002/reproduce.md` and
`outcome.json`. Both readers completed 1,600 steps with real midpoint resume.
Expanded training reduces development tensor error by about 22%, but fails the
fixed gate. No new audio or fresh-test access follows; 96 candidates remain
unlabelled. This control provides no new speech-quality acceptance. Keep all
image-dependent files local, including category labels and per-image metrics.

## Frozen-Qwen acoustic image-reader comparison

`configs/research/acoustic-image-v4.json` prescribes a changed training
objective on the existing canonical development corpus. All fresh candidates
remain reserved. `diagnose_visual_categories.py --dataset <dataset-expanded>
--design <design> --output <new-private-json>` runs in the CPU worker and reports
separability against local training-service labels. It fits only training data;
its classifier never enters speech inference.

With the same worker array as above and a fresh private output root:

```bash
"${worker[@]}" experiment-gpu python scripts/prepare_acoustic_teachers.py --dataset artifacts/private-new/targets/dataset-expanded.json --output-dir artifacts/private-new/acoustic/teachers
"${worker[@]}" experiment-gpu python scripts/train_acoustic_image_reader.py --dataset artifacts/private-new/targets/dataset-expanded.json --teachers artifacts/private-new/acoustic/teachers --initial-run artifacts/private-new/reader-expanded --run-dir artifacts/private-new/acoustic/reader --steps 160
"${worker[@]}" experiment-gpu python scripts/train_acoustic_image_reader.py --dataset artifacts/private-new/targets/dataset-expanded.json --teachers artifacts/private-new/acoustic/teachers --initial-run artifacts/private-new/reader-expanded --run-dir artifacts/private-new/acoustic/reader --steps 320 --resume
```

Teacher generation reads training-supported continuous targets, caches genuine
generated codes and decoded positive controls, and rejects empty/capped
trajectories. Training updates only reader parameters, verifies absent backbone
gradients, and checks checkpoint/teacher/source fingerprints on continuation.
Keep source unchanged between stages. This is a limited scene-repertoire
experiment, not open-ended narration or a trained category-to-text runtime.

Freeze development IDs before fitting. Run the existing `image_probe.py
evaluate` separately for the unchanged baseline and acoustic reader using only
`eval-expanded.json`, identical `--ids`, seed and duration cap. Use different
fresh output directories. `diagnose_acoustic_reader.py` separately takes
`--dataset`, `--teachers`, `--baseline-run`, `--acoustic-run`, `--output`; it
opens supervision for frozen teacher-forced development analysis only. Its
losses cannot establish autoregressive speech quality.

`blind_image_comparison.py` takes `--design`, `--plan`, `--baseline-results`,
`--acoustic-results`, a fresh `--output-dir`, a unique `--run-id` and
`--label-prefix`. It prepares resized EXIF-free local image copies and imports
blinded generated clips without creating ratings. The bounded comparison has
four development images plus training/null controls for each reader. This first
listening comparison does not supply a shuffled-image grounding benchmark or
fresh-test acceptance. Exact exercised commands and mappings stay in ignored
local artifacts; never publish them or the media.

All twelve v4 pairs have now been rated. Private exports and unblinded analysis
are retained under the ignored listening directory; do not alter ratings or
rewrite frozen generation manifests. The canonical baseline has three
development cases described as good Italian and one unclear opening. Acoustic
fine-tuning has no accepted improvement: mean development voice score is
3.25/5 for baseline versus 2.5/5 for acoustic, with equal mean grounding 2.75/5.
These are descriptive scores from one reviewer on four cases per method, not
statistical or fresh-test evidence. The common generic ending was deliberately
authored in every canonical target. Preserve the baseline and reserved fresh
groups; change the supervision/conditioning hypothesis before another run.

## Continuous semantic language-prior research

These scripts are research controls, not the accepted final narration CLI.
Keep the original canonical baseline, frozen image split and all fresh candidates.
Text-feature controls must be labelled text-conditioned diagnostics. All feature
extraction, target projection, fitting and rendering run inside the same managed
worker; only JSON/media preparation runs on the host. Current source/scripts
are bound read-only during development; both dependency locks remain unchanged.

Prepare authored supervision and frozen semantic vectors in separate locked
interpreters, using the worker array above and fresh ignored directories:

```bash
"${worker[@]}" experiment-gpu python scripts/prepare_prior_corpus.py --output-dir artifacts/prior-new/corpus
"${worker[@]}" experiment-gpu /opt/vision/.venv/bin/python scripts/cache_prior_text_features.py --corpus artifacts/prior-new/corpus/corpus.json --output-dir artifacts/prior-new/features
```

The authored split holds out clause/continuation combinations. Selection is
independent of private images and generated speech. Training-only text features
use the pinned Document format and no media tower. The image encoder is unchanged.
`probe_crossmodal_alignment.py` is a diagnostic only; it cannot become a runtime
text-prototype retrieval path.

The whole-prefix v5 flow and causal v6 flow are retained negative experiments.
V6 initializes the causal context/subspace for the next-vector mode objective:

```bash
"${worker[@]}" experiment-gpu python scripts/train_autoregressive_prior.py --corpus-dir artifacts/prior-new/corpus --features-dir artifacts/prior-new/features --run-dir artifacts/prior-new/v6 --steps 2000
"${worker[@]}" experiment-gpu python scripts/train_autoregressive_prior.py --corpus-dir artifacts/prior-new/corpus --features-dir artifacts/prior-new/features --run-dir artifacts/prior-new/v6 --steps 4000 --resume
"${worker[@]}" experiment-gpu python scripts/train_autoregressive_prior.py --corpus-dir artifacts/prior-new/corpus --features-dir artifacts/prior-new/features --run-dir artifacts/prior-new/v7 --initial-run artifacts/prior-new/v6 --recipe multihypothesis --steps 2000
"${worker[@]}" experiment-gpu python scripts/train_autoregressive_prior.py --corpus-dir artifacts/prior-new/corpus --features-dir artifacts/prior-new/features --run-dir artifacts/prior-new/v7 --initial-run artifacts/prior-new/v6 --recipe multihypothesis --steps 4000 --resume
"${worker[@]}" experiment-gpu python scripts/train_autoregressive_prior.py --corpus-dir artifacts/prior-new/corpus --features-dir artifacts/prior-new/features --run-dir artifacts/prior-new/v8 --initial-run artifacts/prior-new/v7 --recipe multihypothesis --modes 32 --robust-semantics --steps 3000
"${worker[@]}" experiment-gpu python scripts/train_autoregressive_prior.py --corpus-dir artifacts/prior-new/corpus --features-dir artifacts/prior-new/features --run-dir artifacts/prior-new/v8 --initial-run artifacts/prior-new/v7 --recipe multihypothesis --modes 32 --robust-semantics --steps 6000 --resume
```

Use fixed source across each continuation. Historical checkpoints require their
recorded source revision; do not bypass fingerprint checks when newer source
changes an objective or signature. Current code reproduces fresh runs with its
own source fingerprints. Exact exercised commands/source revisions are retained
privately under each ignored run directory.

`fit_semantic_image_adapter.py` takes the existing image training dataset/design
and training-text corpus/features. It fits only supported training cases and
persists a numeric coefficient tensor with pairing hashes. It is not a runtime
classifier or caption lookup. `evaluate_autoregressive_prior.py` reads a separate
strict numeric input spec with `--mode image`, optional `--adapter`, prescribed
`--ids` and a fresh `--output-dir`. It checks reader/lock/adapter pairing and
does not read training strings, target prefixes or codes. `--mode
training-text-control` explicitly marks numerical text-feature diagnostics;
`--greedy-candidates` separates a greedy diagnostic from prescribed sampled
continuous candidates. No candidate index has a fixed word/category meaning.

`diagnose_prefix_flow.py`, `diagnose_flow_noise.py` and
`diagnose_mode_prior.py` open cached supervision only for frozen analysis.
Oracle, gold-history and auxiliary ASR controls are not learned-image or human
quality results. Preserve private transcripts/weights/mappings locally and
ingest meaningful real image/audio comparisons for actual human assessment.

The current robust-prior comparison is E6-A through E6-P at
`http://localhost:18080/#m2-prior-v8-a`. `blind_image_comparison.py` accepts
`--candidate-results` (the older `--acoustic-results` alias remains supported),
`--candidate-name` and `--include-shuffled`. The latter adds a complete
derangement of the four new-method development pairs. Six new outputs and six
retained baseline outputs form sixteen pairs because four new audio clips are
deliberately reused on wrong images. Explain these repetitions to the reviewer;
do not count them as independent generations. Existing ratings are preserved,
with no automatic human scores. Fifteen of sixteen pairs have saved ratings. Available notes report good Italian
and variety but weak image relevance, including training. Grounding is not
accepted; all reserved fresh candidates stay excluded.

## Training-group semantic calibration control

Keep the v8 speech checkpoint fixed. Select adapter regularization using only
training scene groups, with no development labels in the selection loop:

```bash
DOCKER_CONFIG="$PWD/.cache/docker-config" .cache/tools/docker compose --profile experiment run --rm -v "$PWD/src:/workspace/src:ro" -v "$PWD/scripts:/workspace/scripts:ro" experiment python scripts/fit_semantic_image_adapter.py --image-dataset artifacts/private-photos-002/targets/dataset-expanded.json --design artifacts/private-photos-002/design/design.json --corpus artifacts/semantic-prior-v5/corpus/corpus.json --features artifacts/semantic-prior-v5/features/features.safetensors --output-dir artifacts/adapter-new/adapter --select-alpha
```

Then use the unchanged numeric-only `evaluate_autoregressive_prior.py` with
`--run-dir artifacts/robust-prior-v8/reader`, the new adapter and existing strict
`eval-expanded.json`; preserve the frozen evaluation-plan IDs and use a new
output directory. The v9 private reproduction file records exact rendered paths
and commands. No additional text features, labels or fresh images are opened.
The current comparison is E7-A through E7-P at
`http://localhost:18080/#m2-adapter-v9-a`, versus unchanged v8 outputs. Four new
clips are intentionally repeated on shuffled images. No ratings are inferred.

Current source tests pass in the existing pinned worker. A full local
`make test-docker` rebuild fails at apt due to Docker disk exhaustion; targeted
unused build-cache cleanup is insufficient to complete a rebuild, and a retry
is cancelled while downloading redundant dependencies. No user images, volumes,
containers or model caches are deleted. Do not describe this as a new worker
image build or a passing full Docker rebuild.

## Specific visual-anchor supervision control

`configs/research/visual-anchors-v10.json` preserves the existing corpus and v8
speech reader. Prepare a private inventory of existing training images only,
with `id`, `path`, `split: train`; do not include development or reserved rows.
The authorized local label service selects one of the existing 60 authored
opening anchors or skips. Its helper always stops the service on completion or
failure, including resumed runs:

```bash
DOCKER_CONFIG="$PWD/.cache/docker-config" .venv/bin/python scripts/label_private_images.py --inventory artifacts/anchors-new/training-inventory.json --output artifacts/anchors-new/anchor-labels.jsonl --limit 239 --mode opening-v10 --docker .cache/tools/docker
```

After confirming the service is stopped, fit numeric calibration in the managed
CPU worker with `fit_semantic_image_adapter.py`, the same dataset/design/corpus/
features arguments as v9, plus `--anchor-labels <private-jsonl> --select-alpha`.
Labels must be unique training IDs with explicit training-only provenance and
validated authored choices. Targets average only training text features sharing
the selected opening. Alpha selection still uses training groups only. The
standalone evaluator is unchanged and reads no labels, target means or strings.
All labels, checkpoint weights and image-dependent diagnostics remain local.
This comparison changes the accepted training subset as well as supervision;
do not attribute any improvement to granularity alone without a matched-subset
control. Frozen numeric anchor-mean controls are text-conditioned diagnostics,
not image-grounding or speech-quality acceptance.

V11 adds `--mode opening-v11` and a `scene_hint` from existing training labels
to each training inventory row. Unsupported hints are skipped. A valid anchor
must belong to that hint's six choices. Malformed model responses are explicitly
excluded with a validation-rejection record; they are never converted to a valid
choice. Resuming preserves completed labels and verifies inventory/prompt hashes.
The service still stops in finally. Supply `--anchor-protocol
configs/research/visual-anchors-v11.json` when fitting this variant. Its coverage
and numerical fit do not establish speech quality or grounding.

## Prior-aware numeric adapter fitting

Use the same cached corpus and accepted v11 training labels. The following
managed-worker commands perform the prescribed midpoint restart. Keep source
fixed across both stages and use a fresh run directory; older runs require their
recorded source revision. The vendor models and v8 speech reader stay frozen:

```bash
worker=(.cache/tools/docker compose --profile gpu run --rm -v "$PWD/src:/workspace/src:ro" -v "$PWD/scripts:/workspace/scripts:ro")
export DOCKER_CONFIG="$PWD/.cache/docker-config"
export LD_CODE_REVISION=$(git rev-parse HEAD)
"${worker[@]}" experiment-gpu python scripts/train_prior_aware_adapter.py --image-dataset artifacts/private-photos-002/targets/dataset-expanded.json --labels artifacts/visual-anchors-v11/anchor-labels.jsonl --corpus-dir artifacts/semantic-prior-v5/corpus --prior-dir artifacts/robust-prior-v8/reader --initial-adapter artifacts/visual-anchors-v11/adapter/adapter.safetensors --run-dir artifacts/calibration-new/adapter --steps 500
"${worker[@]}" experiment-gpu python scripts/train_prior_aware_adapter.py --image-dataset artifacts/private-photos-002/targets/dataset-expanded.json --labels artifacts/visual-anchors-v11/anchor-labels.jsonl --corpus-dir artifacts/semantic-prior-v5/corpus --prior-dir artifacts/robust-prior-v8/reader --initial-adapter artifacts/visual-anchors-v11/adapter/adapter.safetensors --run-dir artifacts/calibration-new/adapter --steps 1000 --resume
```

Then use the unchanged `evaluate_autoregressive_prior.py --mode image` with
strict numeric inputs and the new adapter. Never pass the training corpus or
labels to evaluation. Check actual autoregressive audio; training opening NLL
is not intelligibility or grounding acceptance. All learned weights and
image-dependent outputs remain local. `blind_image_comparison.py --candidate-only`
imports only the new method and its prescribed controls; omit `--baseline-results`.
Retained earlier ratings are a descriptive historical comparison, not a
concurrent blinded baseline or independent replications.

## Opening-specific robustness and semantic-gain comparison

The completed local review is E8-A through E8-J at
`http://localhost:18080/#m2-semantic-gain-v15-a`. It contains ten pairs from six
new outputs, with four deliberate shuffled-image repeats. Do not rerate the six
unchanged v9 baseline clips; any comparison to E7 is descriptive and historical.
No human ratings are created automatically. Assess both Italian clarity and
image connection; likelihood/ASR cannot supply that assessment.

V14 uses the same managed worker array and corpus/feature paths as v8. Add
`--initial-run artifacts/robust-prior-v8/reader --recipe multihypothesis --modes 32
--robust-semantics --opening-semantics` to `train_autoregressive_prior.py`, using
`--steps 1500` then `--steps 3000 --resume` in a new reader directory. The exact
completed source is 9b5f300; preserve source across continuation. Vendor backbones
are not loaded during this cached reader fit. The original v8 fallback remains.

Fit V15 only from training pairs and the completed frozen v14 reader:

```bash
DOCKER_CONFIG="$PWD/.cache/docker-config" .cache/tools/docker compose --profile gpu run --rm -v "$PWD/src:/workspace/src:ro" -v "$PWD/scripts:/workspace/scripts:ro" experiment-gpu python scripts/fit_semantic_gain.py --image-dataset artifacts/private-photos-002/targets/dataset-expanded.json --labels artifacts/visual-anchors-v11/anchor-labels.jsonl --corpus-dir artifacts/semantic-prior-v5/corpus --features artifacts/semantic-prior-v5/features/features.safetensors --prior-dir artifacts/opening-robust-prior-v14/reader --initial-adapter artifacts/visual-anchors-v11/adapter/adapter.safetensors --output-dir artifacts/gain-new/adapter
```

Use `evaluate_autoregressive_prior.py --run-dir artifacts/opening-robust-prior-v14/reader
--inputs artifacts/private-photos-002/targets/eval-expanded.json --mode image
--adapter artifacts/gain-new/adapter/adapter.safetensors --ids q0052 p0055 p0046 q0002 p0003
--output-dir artifacts/gain-new/image-evaluation` in the GPU worker with
`-e LD_REF_TEXT= -e HF_HUB_OFFLINE=1`. Guidance is disabled by default; Qwen
sampling/seed are unchanged. Decoder-specific adapters now verify the frozen
prior hash, preventing accidental mixing of v12 calibration with a newer reader.
The negative pairing control rejects before Qwen loads or any WAV is written.
All media/labels/features/checkpoints and exact private manifests stay local.

The original full worker rebuild remains unverified after Docker storage failure.
About6.265GB of this project's failed dependency layer was subsequently reclaimed
by pruning its unused children before the parent; no user containers, volumes,
model caches or data were removed. The existing pinned worker with source overlays
passes current CPU/CUDA/locked-vision tests; do not claim a newly baked image.


## Image-specific acoustic feasibility control

Read `docs/RESEARCH_REASSESSMENT.md` and freeze
`configs/research/instance-acoustic-v16.json` before preparing data. Select only
existing training rows with distinct scene groups; `label_private_images.py
--mode instance-v16` requires training provenance and shuts down the authorized
local vision container after labeling. Excluded/overlong outputs stay excluded.

In the same pinned GPU worker with read-only source overlays and `LD_REF_TEXT=`:

1. `python scripts/prepare_instance_supervision.py --dataset <existing-development-dataset> --labels <local-training-labels> --output-dir <new-targets>`
2. `python scripts/image_probe.py train --dataset <new-targets>/dataset.json --run-dir <new-baseline> --evaluation-scope development --steps 800`, then the same command with `--steps 1600 --resume`.
3. `python scripts/prepare_acoustic_teachers.py --dataset <new-targets>/dataset.json --output-dir <new-teachers>`.
4. `python scripts/train_acoustic_image_reader.py --dataset <new-targets>/dataset.json --teachers <new-teachers> --initial-run <new-baseline> --run-dir <new-acoustic> --prefix-weight 0 --steps 160`, then `--steps 320 --resume`.
5. Evaluate both readers with `image_probe.py evaluate` on the frozen numeric-only evaluation inputs and chosen IDs. Retain teacher positive WAVs separately.
6. `python scripts/diagnose_instance_acoustics.py --dataset <new-targets>/dataset.json --teachers <new-teachers> --baseline-run <new-baseline> --acoustic-run <new-acoustic> --output <new-diagnostic.json>` measures training oracle, matched and cyclically swapped image codec0 likelihood. It is not a human quality score.

Variable prefix lengths use the existing numeric length head fitted exclusively
to training rows. The acoustic stage trims to each predicted valid length and
requires teacher length parity. Both frozen backbones, all16 codebooks, EOS and
the two-path textless conditioning protocol are preserved. No reserved fresh
images, labels or linguistic strings are loaded at inference. Do not interpret
successful memorization as held-out grounding.
