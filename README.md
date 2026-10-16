# Latent Diaries — image → narrative speech in Italian, no text at inference

**Research bootstrap — not yet a trained image-to-speech model.** Intended for an autonomous coding agent (e.g. Codex) to implement and experimentally validate stage by stage. The local listening service, mock gallery, import/export, GitHub repo bootstrap, HF private upload and safe HF Job launcher exist; actual textless training and image-to-speech generation remain research milestones.

M0 and scoped M1 are accepted: genuine Italian baseline/codec reconstruction,
continuous-conditioning parity, and intelligible speech on two prespecified
fresh numeric-factor combinations. This does not establish image generalization.
Frozen EmbeddingGemma 2 image features and private reader training/resume are
exercised, but the first image-reader pilot fails held-out intelligibility.
The exact-text diagnostic speaks clearly; the development query-reader ablation
does not improve reconstruction. M2 remains incomplete and the final
`narrate-image` command remains fail-closed. Dependency locks, private local
checkpoints and aggregate evidence are recorded in `docs/DECISIONS.md` and
`docs/MILESTONES.md`. No accepted trained image-to-speech model is available.

Human review of all 12 acoustic-comparison clips is complete. The canonical
baseline has good-Italian notes on three of four exposed development cases;
one has an unclear opening. Acoustic fine-tuning does not improve the human
results. The repeated generic ending comes from deliberately constrained
training sentences, so this is limited-repertoire progress rather than original
narration. Fresh test images remain reserved; all private photos, labels,
features, audio, ratings and checkpoints stay local.

E7 review confirms stable Italian and interesting phrases, but weak context:
matched development grounding averages 1.75/5 for both adapter versions. The
training control improves, while fresh-image grounding remains unaccepted.
E8 is also fully reviewed: the user reports no significant improvement, with
only a few acceptable cases. Further template-prior calibration is stopped.
[The reassessment](docs/RESEARCH_REASSESSMENT.md) compares alternatives and
finds that more precise training fits do not establish useful image-to-speech
transfer. Wider readers, bounded last-attention LoRA, direct audio-policy
supervision and generated-history repair are exercised negative controls.
Actual multistep generation/replay parity passes; training-policy accuracy
still degrades on generated histories.

The next comparison preserves the previously Italian-stable continuous language
recipe and expands its training-only narrative corpus: 2,927 utterances with
168 image-specific groups and original-corpus replay. Training, actual checkpoint resumes and image-only generation complete, but
auxiliary speech remains fragmented; this candidate is not promoted. The
Italian-stable reference is retained. Inference reads no captions, linguistic
labels or teacher files. All private artifacts stay local; all 96 fresh
candidates remain reserved. M2 remains unaccepted.

The [independent-review recovery](docs/REVIEW_RESPONSE_20261016.md) corrects the
ambiguous null control and verifies numeric feature provenance. A fixed-checkpoint
oracle ladder produces48 diagnostic WAVs plus four oracle routing controls.
Exact stock/continuous-prefix controls are byte-identical; the first observed
numerical/auxiliary breakdown is inside the continuous prior, before the image
adapter. All18 diagnostic clips are now rated: exact controls remain clear,
while free-prior controls have explicit gibberish notes. Oracle prefixes also
have pronunciation problems. [V25](docs/PROPOSAL_V25.md) remains unstarted.

The [audio-semantic feedback investigation](docs/AUDIO_SEMANTIC_FEEDBACK_LOOP.md)
implements a separate frozen image/audio encoder and passes scoped semantic
feasibility checks, including changed-detail and cross-language controls.
Confirmation covers170 existing teacher utterances; this does not establish
new-image generalization. Pooled cumulative audio loses temporal order, so a
future reader should use the last fragment separately. A bounded comparison
using unchanged Italian-stable checkpoints produces28 blinded pairs,
**AF-01–AF-28**, now fully reviewed. Grounding improves modestly in this eight-image
pilot (1.875→2.625/5), while evocation and surprise remain2/5. The user reports
familiar repeated phrases and persistent context failures. The reranker is not
promoted; no temporal reader is trained. The [next proposed control](docs/ASFL_TWO_FRAGMENT_PILOT.md)
requires genuinely progressive supervision and teacher-story validation before
optimization. M2 remains unaccepted; all private artifacts stay local.

The user authorized [two-fragment corpus preparation](docs/ASFL_TWO_FRAGMENT_PILOT.md).
Twelve original training images yield24 two-fragment sequences and48 genuine
official text-conditioned teacher WAVs. Eight prespecified sequences are in the
gallery as **TC-01–TC-08**, awaiting validation of Italian, image relation and
actual progression. These are teacher upper bounds, not a trained textless
narrator. Media-only features and codec targets are cached; reader training
remains gated on the human corpus review.

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
| Continuous Qwen prefix / teacher-forced assembly / code generation | Tiny CPU/CUDA and pretrained FP32 mechanical controls pass; numeric-conditioned Italian accepted in scoped M1 probe |
| Single-utterance soft-prefix inversion | Resumable training/inference exercised; User accepts 32-token control with codec-0 greedy decoding; one memorized utterance; later sequence reader passes scoped M1 |
| Shared numeric-factor mapper | Blinded review: sampled training case clear; all held-out and null clips fail; no images |
| Direct continuous teacher-prefix regression | Development speech improved, but both fresh combinations lose intelligibility; this affine recipe fails M1.2; no images |
| Sequence-aware numeric reader | 1.20M parameters, 22 training combinations; both fresh tests and training control accepted, null rejected; scoped M1 complete, no images |
| Frozen EmbeddingGemma 2 image-feature cache | Real-image CUDA extraction, normalized 768D vectors, repeat/batch parity exercised; one worker image with separate locked vision interpreter |
| Global image reader pilot | Local private-image training/resume and cached-image-conditioned audio exercised; blinded review rejects all four tested held-out outputs; only training control almost good; M2 incomplete |
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
