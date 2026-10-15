# Milestones — evidence-driven completion

A milestone is complete only with a committed reproduction command, a passing test and run evidence. Progression may stop on a human gate.

2026-10-15 status: M0 and M1 are accepted within the controlled numeric-factor
probe. M0 has genuine Italian baseline speech and codec round-trip evidence.
M1 has frozen-backbone gradient, mask/shift, cache and inference parity tests,
distinct accepted memorized utterances, and now two accepted prespecified fresh
numeric combinations from the v3 sequence reader. The training control is also
accepted; null conditioning is rejected. Standalone generation reads numeric
features and learned weights without variable text, teacher prefixes or codes.

This M1.2 acceptance covers two combinatorial holdouts from 24 authored templates
(22 training), sampled seed 42, one speaker. It does not establish unseen
vocabulary, arbitrary stories, seed robustness, image dependence or artistic
quality. The matched affine baseline does not pass fresh intelligibility; earlier
shared-MLP and v2 affine failures remain documented in DECISIONS. All reviewed
holdouts are now exposed and must not become fresh tests for a revised recipe.
See `validation/2026-10-15-sequence-reader.json` and OPERATIONS for evidence and
reproduction. Individual listening notes and media remain private.

M2–M5 remain unachieved. A global image reader has now actually trained/resumed
on 50 private images, with 12 group-disjoint holdouts. Four held-out audio probes
and train/null controls form 13 local blinded pairs including shuffled-image
assignments. Human review is complete: only the training control is almost good; all four
tested held-out outputs fail intelligibility. This pilot is a negative M2 result.
The exact-text diagnostic now has human acceptance for pronunciation/prosody.
A fixed development-only query-reader/ridge-length ablation also fails to improve
reconstruction; it produces no new audio and establishes no fresh generalization.
A subsequent canonical-supervision control trains/resumes on 46 original-cohort
versus 239 expanded images, with 55 shared development images. Equal-length
sentence targets remove length prediction; broader data reduces development
prefix MSE by 22%, but the precommitted gate still fails. No new audio is
generated, and 96 fresh candidates remain unlabelled. This bounded repertoire
does not establish open-ended narration or M2 acceptance.
An acoustic-loss follow-up now completes 320 reader updates through frozen
Qwen, with real midpoint resume. A linear visual diagnostic predicts local
category labels on development images with 84% accuracy, versus 20% majority
and 18% shuffled-label controls. Development codec0 teacher-forced CE improves
by 4.7%, while residual CE and prefix error slightly worsen. Twelve blinded
development clips (E5-A through E5-L) await human listening; these diagnostics
do not establish intelligibility or fresh-image grounding. Final
`narrate-image` CLI remains fail-closed; research training is exercised. EmbeddingGemma 2 compatibility is now exercised in a separately locked interpreter
inside the same worker image: two real images produce normalized 768D vectors
with repeat/batch parity. The cached image-to-reader training link is exercised; grounding acceptance
remains open. BF16 cache parity is also open; accepted speech uses FP32 Qwen.

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
