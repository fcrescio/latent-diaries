# Milestones — evidence-driven completion

A milestone is complete only with a committed reproduction command, a passing test and run evidence. Progression may stop on a human gate.

2026-10-14 status: M0 baseline and genuine-speech codec round-trip are accepted
by the user. Continuous assembly, frozen-backbone gradients, cache parity and
checkpoint inference are exercised. M1.1 remains **failed for intelligibility**:
the user rejected both sampled and fully greedy run-002 audio despite acceptable
voice/prosody. Matched-target CE, teacher-distribution KL, generated-history
supervision and a 32-token capacity control are documented experiments, not
speech-quality successes. Low teacher-forced loss does not survive generated
histories. The stock teacher-derived prefix reproduces its baseline exactly but
is only a positive protocol control. M1.2 and M2–M5 remain unachieved. BF16 cache
parity and EmbeddingGemma 2 compatibility remain open technical questions.

| Milestone | Tasks | Evidence / exit gate |
|---|---|---|
| M-1 bootstrap | env, GitHub remote, HF private repos, Docker and review UI | `make test`, `make test-docker`, local browser, no tracked private files |
| M0 baseline | use `qwen-baseline` with opt-in reference audio; tokenize round-trip; measure runtime | `artifacts/m0_italian.wav`, real spoken Italian, latency+hardware manifest |
| M1.0 train assembly | `SoftConditioningBuilder` and tests for 1024D, EOS, masks, label shift, KV cache | unit tests with tiny synthetic tensors and integration test against pinned Qwen model |
| M1.1 inversion | learn per-utterance 16/32 soft tokens targeting a known Italian codec sequence | distinct intelligible textless Italian outputs, not a fake demo or copied source audio |
| M1.2 generalization | condition on learned latent inputs unseen during training; compare null prefix | generation changes nontrivially and remains intelligible; analyze collapse |
| M2 reader global | freeze EmbeddingGemma2, precache global embeddings; paired image/audio trainer | held-out images -> real Italian audio and private checkpoint; no inference text |
| M3 reader sequence | inspect visual feature API; compare global vs seq reader; optional LoRA | ablation table: intelligibility, image dependence, RAM, artistic listening |
| M4 art | single consented expressive narrator; curated photos/recordings; blinded review web | 20+ listening pairs, user ratings and commentary, small gallery with provenance |
| M5 final prototype | multi-frame memory; optional wearable adapter; CLI end-to-end, private release | `latent-diaries narrate-image` outputs one genuine WAV from one or more new images, no captions; full README/setup/checkpoint HF link |

## Regression suite (must build over time)

- Stock Qwen output parity with pinned package in non-streaming mode before applying reader.
- EmbeddingGemma vision-only modality config loaded correctly; no silent fallback to unrelated CLIP encoder.
- Nonzero reader grads, frozen Qwen grads remain absent when expected.
- Same-frame audio target leak prevention; no variable text embeddings during final inference.
- EOS on valid duration; 16 codebooks predicted / decoded; deterministic seed controls.
- Shuffled-image, null-image and held-out-walk probes; blind identity preserved.
- Privacy: no .env in Docker image, git, evaluation exports, HF checkpoint; API Basic auth enforced.
- HF upload private repo, metadata includes upstream revisions/licenses, seed, train manifest digest, code commit and dependency lock.

## Advancement rules

If M1 continuous conditioning is impossible without extensive retraining, try late-layer LoRA after a documented negative result; do not abandon the no-text inference objective without discussing a change in project identity with the human. When local GPU unavailable, prepare remote job but require explicit G1 approval. If quality gate depends on artistic judgment, present actual samples in review service; wait for the user ratings before claiming success.
