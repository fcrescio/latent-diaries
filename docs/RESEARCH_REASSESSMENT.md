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
