# Milestones — evidence-driven completion

A milestone is complete only with a committed reproduction command, a passing test and run evidence. Progression may stop on a human gate.

2026-10-15 status: M0 baseline and genuine-speech codec round-trip are accepted.
Continuous assembly, frozen-backbone gradients, cache parity and checkpoint
inference are exercised. M1.1 memorization acceptance is satisfied by two distinct
human-accepted Italian outputs: run-006 mixed decoding and shared-mapper-001
sampled training case c01 (blind C). The first accepted control is
run-006, 32 learned continuous tokens, greedy codec-0 and sampled residuals,
seed 42. Fresh CLI inference reproduces that accepted waveform exactly without
reading text or teacher targets. The original fully sampled run-006 has an
unclear middle; run-002 sampled/greedy outputs were rejected. Acceptance is
specific to the successful decoding control, not all checkpoints/settings.
The user also finds run-007 fully sampled speech almost good, with only half a
word unintelligible. This is a second distinct, mostly intelligible memorization
control; its greedy version has no listening acceptance. M1 remains incomplete:
full clarity across utterances and unseen-input generalization are unverified.
A shared 4.23M-parameter numeric-factor mapper has now trained on six prespecified
combinations, with two held out. Held-out teacher-forced agreement is roughly
level with null conditioning. Blinded human listening accepts only the sampled
training case; both held-out inputs fail in both decoding modes. This probes
compositional generalization, not image grounding. An additive audio-KL control
did not help. Human listening accepts the direct-prefix training control and one
exposed development case, finds the other almost clear, and rejects null. A fresh
v2 combinatorial test was frozen before new teacher generation/fit and has now
completed blinded listening: training speech is accepted; both fresh combinations
start clearly then become unintelligible, and null loops in gibberish. The stock teacher-derived
prefix is a separate positive protocol control.
A v3 sequence-aware reader and affine baseline have now trained on 22 combinations
with two new tests. Sequence reconstruction error is much lower, but the eight
blinded comparison clips await human listening. M1.2 remains incomplete.
M1.2 failed under audio-KL training; direct-prefix supervision improves exposed
development cases but fails the fresh test; M2–M5 remain unachieved; BF16 cache parity and EmbeddingGemma 2
compatibility remain open technical questions.

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
