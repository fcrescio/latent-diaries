# Research reassessment after E8

The user reports no significant improvement. M2 remains unaccepted. We stop
optimizing the continuous template language prior and its semantic gain.

## What the evidence supports

E8 has ten rated pairs: four matched, four shuffled, one training and one null.
Matched grounding averages 2.75/5 versus 1/5 for shuffled pairs, but one matched
utterance is explicitly rated gibberish, and the user finds only a few OKish.
This is a tiny, repeatedly exposed development set, with no concurrent blinded
comparison against E7. Neither the average nor auxiliary ASR establishes an
improvement. Voice ratings also vary across repeated audio paired to different
images; they are human judgments, not an objective intelligibility metric.

The current prior learns from 60 authored openings and 40 authored tails.
Many combinations do not add new linguistic structures or image details. The
image adapter first compresses image variation into averages of those targets;
a second small autoregressive model then generates projected text vectors;
finally Qwen generates audio autoregressively. Each stage can lose grounding,
and both generation stages can accumulate error. Training loss has repeatedly
failed to predict human outcomes. More gain, candidate or routing tuning is not
our best next test.

Stock text-conditioned Qwen can produce clear Italian. Existing global image
vectors support a coarse train/dev classifier (83.6% development accuracy),
although this says nothing about fine object relations. These controls direct
attention first to supervision and alignment, rather than replacing backbones.
The earlier frozen-Qwen acoustic control used only ten canonical sentences,
320 steps and a prefix reconstruction regularizer. Its failure does not rule out
acoustic learning from image-specific utterances.

## Alternatives and decision

| Path | Benefit | Limitation / decision |
| --- | --- | --- |
| More prior/gain tuning | Cheap and preserves current runtime | Narrow target repertoire, repeated negative human evidence: stop |
| Image-specific targets and direct acoustic supervision | Optimizes the actual speech decoder; preserves textless inference and frozen backbones | Needs genuine teacher audio; first isolate memorization before scaling: do now |
| LoRA on final Talker attention layers | Could adapt Qwen to visual prefixes instead of forcing all adaptation into reader | Added capacity can memorize; only after frozen acoustic feasibility control, with parameter/VRAM accounting |
| Intermediate visual tokens and resampler | Could preserve spatial details lost by global pooling | Requires verified upstream tensor semantics and evidence that pooling is the bottleneck; defer |
| More training examples only | Needed for eventual coverage | Does not fix an alignment mechanism that cannot fit a small distinct set; defer scaling until feasibility |
| Caption then TTS at runtime | Useful separate diagnostic baseline | Uses text at inference and does not satisfy the mission |

V16 uses 24 distinct existing training groups, selected before local vision
labeling, with short image-specific Italian utterances mentioning two visible
details. No fixed ending. A small global reader first fits their numeric
prefixes; then it receives genuine codec targets through frozen Qwen, with no
prefix-MSE penalty in the acoustic stage. Keep the warm-start control and exact
teacher speech. Resume both fitting stages from real midpoint checkpoints.
Compare matched, swapped and null image behavior, and inspect training versus
previously exposed development generation separately. This is a feasibility
experiment, not evidence of new-image generalization.

The next decision depends on the failure location: teacher speech failure means
repair supervision; inability to reproduce distinct training utterances means
alignment/optimization needs work (including a bounded LoRA comparison);
training success with development failure means prioritize data coverage and
regularization. Do not spend the 96 reserved fresh images on this diagnosis.
Human quality judgments remain required; no automated score will approve M2.

Exact protocol: `configs/research/instance-acoustic-v16.json`. All supervision,
media, instance outputs and checkpoints remain local. No paid jobs or uploads.

## V16 result and immediate consequence

The bounded control completes both real resumes (prefix800/1600,
acoustic160/320). Seven image-specific pairs pass validation from24 candidates.
All four prescribed training images reproduce their own genuine teacher WAVs
**byte for byte**, after extending two truncated controls to160 frames. This
establishes reproduction through the image-only interface, not human acceptance
or new-image generalization.

Training codec0 CE is0.80752 for oracle and fitted image prefixes, versus2.28395
for cyclically shuffled images. Acoustic fitting reduces matched CE to0.68430
(15.3%) while shuffled CE remains2.28587. Nevertheless, all four development
outputs reach the105-frame cap under both readers, and auxiliary transcriptions
remain fragmented on three cases. These diagnostics do not establish human
quality, but provide no basis to promote the new model or request another full
artistic review. Keep the earlier Italian prior as a fallback.

This small fit is deliberately overparameterized (5,119,616 reader parameters)
and cannot adjudicate generalization mechanisms. It rules out a basic inability
to learn distinct image-to-speech mappings in the verified frozen interface.
The next action is image-specific supervision across the existing239 training
rows, under `configs/research/instance-supervision-v17.json`, not another gain
sweep or immediate LoRA expansion. Prior synthetic combination counts should
not be mistaken for image-specific training coverage. V17 prepares data only;
its labels are untrusted training annotations, not human judgments.

Installed vision-source inspection confirms `get_image_features` runs the
vision tower, applies `embed_vision` and splits by nonpadding patch counts;
`forward` exposes the projected features before language-model fusion as
`image_hidden_states`. They are not interchangeable with final normalized768D
vectors. A future visual-token ablation must exercise dimensions, masking,
position semantics and batch parity before training; source availability alone
is not a validated capability.

## Main pivot: train visual prefixes in the speech domain

The wider word-vector reader and its matched last-attention LoRA variant both
retain fragmented development ASR. More capacity at that same representation
has not established transfer. We therefore stop using per-image sentence-vector
reconstruction as the main learning problem.

The new reader emits32 continuous positions for every image. Its output begins
at a shared, previously learned M1 language prefix; the image-dependent output
weights start at zero. The anchor is numeric model initialization, never a
transcript or per-image target. Both frozen-Talker and last-four-attention LoRA
runs use the same initialization. They learn from audio-code distributions only;
there is no predicted sentence length or sentence-vector reconstruction loss.
A constant repeated anchor is explicitly a grounding failure.

The M1 history also matters: matched hard CE over a sampled codec trajectory
had diverged from teacher distributions, whereas KL plus greedy codec0/sampled
residuals produced the human-accepted memorization control. The direct visual
runs restore that working recipe. A frozen teacher supplies aligned codec0
(including EOS) and all15 residual-head logits on genuine gold previous audio
codes. Its training-only text-derived prefixes are used to CACHE teacher
policies, not as student inputs or vector targets. Student training loads only
image vectors, audio codes, detached policy targets and its shared numeric
initialization. Inference loads none of the teacher artifacts.

The first1600-step direct pair still underfits training speech. Before launching
another fit, a four-training-image optimizer probe compares physical prefix
updates from the same initial function. The successful M1 free-prefix step
moves0.029997 RMS, while the inherited normalized reader moves0.000725 RMS,
about41 times less. Unit output scaling with a1e-3 final-head rate moves0.035667
RMS and decreases mean training KL2.04456→1.70936. This is a training-only
parameterization calibration, not a quality metric or development selection.

The calibrated3200-step comparison uses this output scale and head rate, with
1e-4 on the remaining reader and optional LoRA, actual1600 resume, identical
fresh anchor initializations, and the same170 cached teacher policies. Recent
training codec0 KL reaches0.02–0.04; generated speech still requires its own
assessment. No loss threshold approves Italian or M2.

A separate CPU-only feature-geometry probe finds that training-centering the
input affine could reduce initial head-gradient cosine0.612→0.140 and design
condition number249→47 without changing the initial anchor output. This remains
a deferred initialization alternative, not an exercised speech improvement.
Do not keep adding changes before inspecting the prescribed generation results.
