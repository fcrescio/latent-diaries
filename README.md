# Latent Diaries — image → narrative speech in Italian, no text at inference

**Research bootstrap — not yet a trained image-to-speech model.** Intended for an autonomous coding agent (e.g. Codex) to implement and experimentally validate stage by stage. The local listening service, mock gallery, import/export, GitHub repo bootstrap, HF private upload and safe HF Job launcher exist; actual textless training and image-to-speech generation remain research milestones.

Implementation has started: `uv.lock` pins the dependency set. Continuous Qwen
conditioning, teacher forcing and code generation pass tests using tiny real
Qwen classes on CPU and CUDA (16 tests pass in the GPU worker). M0 has generated
an Italian baseline and genuine-speech codec round-trip locally; the user
confirmed clear Italian and an acceptable voice. Pretrained FP32 assembly and cache controls also pass.
The user rejected run-002 sampled and greedy speech as unintelligible.
The user accepts the 32-token run-006 with greedy first-codebook decoding as
fully intelligible Italian. A second separately learned utterance, run-007 with
sampling, is almost clear per the user, with half a word unintelligible. These
are memorization controls; generalization and image conditioning remain
unverified. M1 is incomplete.
No trained image-to-speech checkpoint exists. See `docs/DECISIONS.md` for evidence.

## Artistic brief

Turn opt-in photos from a walk/wearable into an intimate, slightly unreliable oral memory in Italian. Images become a condition for a voice to speak, not an instruction to caption a scene. The work has its own aesthetic judgments (evocation, grounding, voice, surprise), not ASR WER as its aim.

```
photograph
   | EmbeddingGemma 2 (frozen, vision+text)
   v
   small narrative reader (trainable, continuous K x 1024 tokens)
   v
   Qwen3-TTS 0.6B Base, Italian, fixed speaker, no text tokens in inference
   v
   12Hz 16-codebook codec tokens -> Qwen audio decoder -> Italian voice
```

**Technical caveat:** Qwen3-TTS is originally a text-conditioned synthesizer, **not** a textless speech language model. Continuous visual conditioning and generation of *new intelligible Italian speech* is an unproven hypothesis. The agent must implement/ablate this and fail honestly when it does not work. See `docs/ARCHITECTURE.md`.

## Quick start (operator)

Prerequisites: Docker Compose v2, Python >=3.11, git, `gh` CLI, and GPU runtime if local CUDA available.

```bash
cp .env.example .env
chmod 600 .env
# Edit .env locally: GH_TOKEN, HF_TOKEN, GITHUB_OWNER, HF_NAMESPACE,
# REVIEW_PASSWORD.  Tokens are never placed in GitHub or HF datasets.
uv sync --locked --extra test
make test
make init
make demo         # SYNTHETIC gallery for UI verification; NOT model output
make review       # open http://localhost:8080, authenticate with REVIEW_USER/PASSWORD
```

Then, after reviewing `.gitignore` and permissions:

```bash
make github       # git init, create private GitHub repo, commit and push via GH_TOKEN
make hf-create    # create PRIVATE HF checkpoint + PRIVATE dataset repos
make test-docker  # runs container smoke checks, may download large ML dependencies
```

If you have an explicitly consented Italian reference WAV, put it under `run-data/voices/voice.wav` and set `LD_REF_AUDIO=/workspace/run-data/voices/voice.wav` in `.env`. Then `make m0` executes genuine Qwen TTS in the worker container. For local NVIDIA GPU: `make gpu-doctor`; M0 can be invoked via `docker compose --profile gpu run --rm experiment-gpu latent-diaries qwen-baseline`. Beware the GPU PyTorch wheel and driver must work together; `doctor` reports availability. On Apple Silicon, standard Docker does not provide Metal/MPS to Linux containers.

## Working with Codex

Start Codex from repository root and tell it:

> Read AGENTS.md first, then docs/ARCHITECTURE.md and docs/MILESTONES.md. Execute the milestones sequentially until a human gate blocks further progress. Implement tests, commit/push reproducible changes, publish reviewed checkpoints privately to Hugging Face, and prepare blinded samples in the local review UI. Never charge GPU resources or publish images/audio publicly without my explicit approval. Do not claim training success from demos or scaffolds.

The agent can implement, test, debug and publish code, while you retain control over secrets, cloud spending, privacy and artistic evaluation. It cannot complete an experimental ML project without actual compute, appropriate data and human feedback; milestones explicitly capture those dependencies.

## Current bootstrap capabilities

| Component | Bootstrap status |
|---|---|
| Source tree, Python package, CI, Docker worker, Docker review UI | Provided; local code tested |
| Synthetic demo, validated media importer, SQLite ratings, auth, CSV export | Implemented |
| HF private repositories | Created under `fcrescio`; no data or checkpoint uploaded |
| Checkpoint upload, guarded HF Jobs launcher | Entrypoints provided; **not exercised** with real checkpoint/jobs |
| Official Qwen Italian TTS baseline | GPU run produced 5.44 s audio and codec round-trip; user accepted speech/voice |
| Continuous Qwen prefix / teacher-forced assembly / code generation | Tiny CPU/CUDA and pretrained FP32 mechanical controls pass; learned textless speech pending |
| Single-utterance soft-prefix inversion | Resumable training/inference exercised; User accepts 32-token control with codec-0 greedy decoding; one memorized utterance; M1 incomplete |
| Shared numeric-factor mapper | Blinded review: sampled training case clear; all held-out and null clips fail; no images |
| Direct continuous teacher-prefix regression | Development speech improved, but both fresh combinations lose intelligibility; M1.2 incomplete; no images |
| Visual reader / image-conditioned training | **Not implemented yet** |
| Genuine Italian image-to-speech checkpoint / wearable experience | **Not yet achieved** |

Use `docs/capabilities.json` for machine-readable safety flags. Never flip flags without tests.

## Documentation index

- [`AGENTS.md`](AGENTS.md): autonomous agent operational rules and hard gates.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md): tensor shapes, prompt replacement, Qwen codec invariants and test design.
- [`docs/MILESTONES.md`](docs/MILESTONES.md): actionable tasks M-1 to M5 and acceptance criteria.
- [`docs/OPERATIONS.md`](docs/OPERATIONS.md): local/cloud/GitHub/HF commands and deployment.
- [`docs/HUMAN_GATES.md`](docs/HUMAN_GATES.md): authorization, private data and human review.
- [`docs/DATASET.md`](docs/DATASET.md): paired image/speech manifests and consent/provenance.
- [`docs/DECISIONS.md`](docs/DECISIONS.md): dated engineering log for the agent.

## Official upstream references

- https://huggingface.co/google/embeddinggemma-2
- https://huggingface.co/docs/transformers/model_doc/embedding_gemma2
- https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base
- https://github.com/QwenLM/Qwen3-TTS/blob/main/qwen_tts/core/models/modeling_qwen3_tts.py
- https://huggingface.co/docs/hub/jobs-configuration
- https://huggingface.co/docs/huggingface_hub/guides/upload

## Privacy and licensing

Default GitHub repository and HF repositories are private. Third-party models remain referenced by upstream IDs, not republished in checkpoints. Raw photos, recorded voices and listening feedback stay in ignored local directories unless a *private* HF dataset upload is separately approved. The review server binds loopback only, has HTTP Basic auth and a CSRF token, and is **not suitable for public internet exposure** without additional protections. Follow all model/data licenses and applicable photo/audio privacy requirements.

Apache-2.0 license applies only to this bootstrap's original code, not to third-party model weights/data.

For a prewritten agent kickoff instruction see [`docs/CODEX_LAUNCH.md`](docs/CODEX_LAUNCH.md).
