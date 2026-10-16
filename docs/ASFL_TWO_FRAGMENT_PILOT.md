# Two-fragment pilot: proposed next control, not a training authorization

The completed AF review shows a modest selection benefit in grounding, no gain
in evocation or surprise, and persistent familiar phrases. Reranking cannot
supply concepts or narrative transitions absent from its candidate pool.
Audio-encoder semantic feasibility is useful infrastructure, not proof that a
generator will use feedback or acquire a richer repertoire.

The next hypothesis concerns **generation and supervision**, not a larger pool,
new seed sweep, tuned cosine threshold or enlarged V25 prior. Test whether a
small reader can use the last spoken fragment to choose a genuinely different,
image-grounded continuation while keeping Italian clear.

## Corpus upper bound before optimization

Use twelve already-authorized training images from twelve distinct capture
groups. Choose identities and an eight-image training/four-image diagnostic
validation split before generating or scoring material. Cover the ten historical
scene categories, with the two extra identities chosen by a fixed hash rule.
Do not access the96 reserved images or reuse development listening to select
training examples. This tiny split is diagnostic, not a generalization claim.

For each image, prepare **two distinct narrative branches**, each with two short
fragments:24 sequences,48 teacher utterances. A1 introduces a specific visual
detail or perspective; A2 adds an event, association or change in perspective
that depends on A1. It must preserve the image connection and make sense as a
continuation. Avoid fixed endings, category-only sentences and synonymous
rewrites. Two branches for the same image make audio history informative rather
than allowing the image alone to identify a unique target.

First inspect and validate the narrative upper bound: if the proposed teacher
stories are themselves repetitive, weakly grounded or merely captions, reject
the corpus before spending resources on a reader. Use the authorized local VLM
only for drafting/labeling; stop its container before loading other GPU models.
Machine judgments are fallible and cannot provide human approval.

Generate teacher speech with the separate official text-conditioned baseline,
using the consented voice. Text and linguistic labels remain training-only.
Review a prespecified sample of eight full sequences, covering both branches
where possible and both split partitions, for Italian clarity, image connection
and actual progression. Human subset validation is required before training.
Publish neither these private stories nor any media. Freeze provenance, targets,
capture groups and the review outcome before optimization.

## Minimal comparison after the corpus gate

Compare an image-only reader with an image-plus-last-audio reader. Keep Qwen and
EmbeddingGemma frozen, the speaker and synthesis protocol identical, and the
reader capacity/budget comparable. Use the verified continuous prefix assembly
in both initial and trailing conditioning paths. Teacher-force gold previous
audio codes and verify gradients, masks, EOS and inference parity.

For the same image and fixed synthesis seed, test:

- Matched A1 audio from the intended branch.
- The other branch's A1 **from the same image**. Cross-image shuffle alone cannot
  separate history dependence from visual category cues.
- An A1 from a different image, retaining the original permanent image anchor.
- Explicit absent history with a presence mask, not a silent WAV.

Encode the last fragment separately; pooled cumulative audio is not ordered
memory. Keep image and audio features separate initially and omit native joint
features or recurrent memory ablations until this two-fragment control works.
Evaluate teacher histories first, then actual generated A1 histories. Never use
captions, transcripts, ASR, retrieved linguistic tokens or a textual plan to
condition the student's inference path.

## Outcomes that decide whether to continue

Keep intelligibility, grounding, progression, repetition and branch dependence
separate. A different waveform or higher cosine alone is not a continuation
improvement. Human comparison must show an audible advantage over image-only
generation, with no systematic loss of Italian or image connection. A reader
that ignores the within-image history swap has not demonstrated useful feedback;
a reader that merely repeats A1 has not demonstrated progression.

If the text-conditioned upper bound is good but continuous students lose
Italian even on training controls, investigate the continuous-to-speech bridge
before enlarging the temporal reader. If training controls work but the four
diagnostic validation images fail, report memorization and stop scaling.

This document prepares a bounded experiment. No new sequential corpus,
teacher audio, reader training or paid job has been started by writing it.
