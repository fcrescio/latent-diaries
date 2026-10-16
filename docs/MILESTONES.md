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
development clips (E5-A through E5-L) have now been reviewed. Baseline notes
indicate good Italian on three of four development cases, with one unclear
opening. Acoustic fine-tuning does not improve the human results. The repeated
generic ending is imposed by the authored control sentences; narrative variety
remains untested. No fresh-image or shuffled grounding acceptance follows. Final
`narrate-image` CLI remains fail-closed; research training is exercised. EmbeddingGemma 2 compatibility is now exercised in a separately locked interpreter
inside the same worker image: two real images produce normalized 768D vectors
with repeat/batch parity. The cached image-to-reader training link is exercised; grounding acceptance
remains open. BF16 cache parity is also open; accepted speech uses FP32 Qwen.

A continuous semantic-language-prior follow-up uses authored Italian training
sentences and training-only text embeddings. Whole-prefix and causal flows fail
word-content controls. A continuous-candidate autoregressive reader recovers
coherent auxiliary text-control transcriptions but initially fails image transfer.
The subsequent robust-conditioning recipe completes 6,000 updates with real
midpoint resume, then generates all four existing development cases, training
and null controls without reading text/teacher targets at inference. Its 16-pair
E6 gallery includes the retained baseline and shuffled-image assignments.
Fifteen of sixteen E6 pairs now have actual ratings. Available new-method notes
report good Italian and variety, but weak image relation including training.
Grounding is not accepted; fresh candidates remain reserved and M2 is incomplete.
A training-group-calibrated adapter control now isolates semantic shrinkage while
keeping the speech reader frozen.

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

E7 is fully reviewed: Italian is stable and phrases are interesting, but matched
grounding averages 1.75/5 for both compared adapters. Fine visual-anchor training
supervision, prior-aware calibration and bounded guidance controls are exercised
without fresh access. The opening-robust reader completes3000 steps with real
resume; a scalar semantic gain selected from training-group validation is folded
into its frozen image adapter. E8-A through E8-J contains six genuine new WAVs
in ten matched/shuffled/training/null pairs. Review is complete; the user finds no significant improvement. Further prior
calibration is stopped in favor of the image-specific acoustic feasibility
control specified in `configs/research/instance-acoustic-v16.json`;
no M2 acceptance or unrestricted narrative-generation claim follows. All 96 fresh
candidates remain unlabelled/unencoded; private media, labels and weights stay local.


The subsequent reassessment prepares170 image-specific training utterances,
corrects a measured width128 linear-prefix rank restriction, and exercises a
width256 acoustic comparison with genuine targets and actual resumes. Training
fits and losses improve while development auxiliary transcriptions still fail.
Final-attention rank8 LoRA is therefore tested under a separately frozen protocol;
its original vendor tensors remain frozen. M2 and fresh generalization remain
unaccepted pending actual speech/grounding evidence and human assessment.

### 2026-10-16 critical reassessment (M2 still incomplete)

E8 human review reports no significant improvement. Repeated development
exposure does not justify a fresh-test claim. Image-specific acoustic fitting,
a wider reader, last-attention LoRA, direct frozen policy distillation and two
student-history repair iterations are exercised without demonstrated quality
improvement. Actual24-step generation/replay parity passes within3.2e-5; low
gold-history loss is insufficient evidence for intelligible generated speech.

V24 instead repeats the previously Italian-stable continuous-language curriculum
on a richer training-only narrative corpus, with original-corpus replay and
training-group-CV image adaptation. The prepared corpus has2927 utterances and
168 usable training-image groups. Its490 language-held-out rows are paraphrases
of training images, not held-out-image evidence. No quality or grounding gate
is passed by preparation or loss. All96 reserved images remain untouched.

V24 training/resume and six standalone image-only generations complete.
Auxiliary results remain fragmented, with1/4 development clips reaching the
160-frame cap. No gallery or human approval is created; the candidate is not
promoted. The original Italian-stable baseline is retained. Full CUDA suite
65passed/2skips; worker CPU62passed/4skips and vision4passed. M2 remains open.

### Independent-review recovery (2026-10-16)

Confirmed affine-null and numeric-input provenance defects are corrected with
regression tests; historical results stay unchanged. The no-training oracle
ladder produces48 genuine WAVs across three seeds, plus four offline
routing/candidate controls. Six exact stock/continuous-prefix pairs are
byte-identical. Continuous-prior drift is already evident with exact training
sentence semantics, before image adaptation. See `REVIEW_RESPONSE_20261016.md`
for limits and processed-policy analysis. Eighteen blinded diagnostic clips
OD-A–OD-R await human listening; these are not an approved image-only candidate.
V25 remains a single conditional proposal and is not started. M2 is unaccepted,
`narrate-image` stays fail-closed and all96 fresh images remain reserved.


2026-10-16 audio-semantic follow-up: OD listening is complete. Exact controls
remain clear; free-prior and oracle-prefix pronunciation problems persist.
The new isolated frozen media encoder passes FP32 image parity and scoped
audio semantic feasibility, including an additional140 existing-teacher check
and cross-language diagnostic. A no-training28-pair comparison using unchanged
Italian-stable checkpoints is in the loopback gallery as AF-01–AF-28, pending
human assessment. Cumulative audio embedding loses temporal order. No temporal
reader/corpus is trained, no reranker is promoted, V25 is unstarted, and M2
remains unaccepted. All96 fresh images and all private media remain reserved
or local respectively. See `AUDIO_SEMANTIC_FEEDBACK_LOOP.md` and its aggregate
validation JSON for evidence and limitations.


Completed AF human outcome: all28 pairs rated. Matched max-cosine grounding
improves descriptively from1.875 to2.625/5 on eight existing development images,
while evocation/surprise remain2/5. The user reports persistent familiar phrases
and context failures. The reranker is not promoted and no temporal feedback
competence is established. Keep encoder feasibility separate from narrative
quality. `ASFL_TWO_FRAGMENT_PILOT.md` proposes validating genuinely progressive
teacher stories before optimization; no new sequential corpus or training is
started. M2 remains unaccepted.
