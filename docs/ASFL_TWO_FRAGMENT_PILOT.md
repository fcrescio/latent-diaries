# Two-fragment pilot: authorized corpus preparation, training gated

The user authorized preparation of this pilot after the AF assessment. The
selection, drafts and editorial corpus now exist locally. Text-conditioned
teacher synthesis and the prespecified eight-sequence listening subset implement
the corpus-quality upper bound. This authorization does not waive the required
human corpus validation before reader training.

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

## Executed preparation and interpretation

The source inventory is the original170-image accepted training cohort. Twelve
images from twelve capture groups cover all ten historical scene categories;
the 8/4 image split and eight review identities are frozen before drafting.
Two branches per image produce 24 sequences and 48 fragments. The internal
validation partition is drawn from original training images and is not a fresh
test set. All96 reserved images remain untouched.

The user-authorized local vision container drafts from selected pixels with
stripped metadata. Its reported identity is recorded locally, not assumed to be
an independently verified upstream revision. Raw model output is untrusted data.
The drafts omit required branch labels/use a different object shape; preserve
them and normalize the schema only through explicit editorial records.
Assistant inspection of the selected pixels also removes unsupported objects
and rewrites weak continuations. These authored first-person microevents are
artistic imaginings, not factual claims about the photographer's life.

Two sealed editorial/corpus versions and the machine audits are retained.
One bounded revision addresses weak progression and same-image history
ambiguity on six images. The correlated machine judge also makes questionable
criticisms, including treating an imagined action as contradicted merely
because an object is at rest in the photograph. Its flags are diagnostic,
not truth labels or human approval. Do not repeatedly rewrite until a machine
score passes; the prespecified human subset decides corpus acceptance.

Teacher speech uses the separate **official text-conditioned** Qwen wrapper,
fixed consented voice, Italian language marker and no reference transcript.
It is not a learned image-conditioned or textless output. The first greedy
four-item batch stops at the fourth item's validity guard; three outputs are
retained. The original guard combines invalid/empty/capped conditions and the
fourth tensor was not saved, so its exact failure subtype is not retrospectively
claimed. New synthesis code persists capped batch results before stopping.
Isolated greedy/sampled controls on that fragment both terminate normally.
Batch composition and residual RNG coupling are not separately identified.
Return globally to the previously accepted official M0 single-utterance sampled
baseline, seed 42 per utterance,240-frame bound and no per-case retries. Preserve
the failed run, controls and explicit recipe amendment separately.

For listening, each selected A1/A2 sequence contains its two genuine WAVs with
a 0.45-second pause. The sequence identities were selected before any story or
audio existed. Retain branch/source mapping privately and create no votes.
The run is `asfl-teacher-corpus-20261016`, labels **TC-01–TC-08**. These are
teacher-story upper bounds, not evidence that a reader has learned a memory.

Rate image relation, Italian/voice, narrative force and surprise using the
existing gallery. In the free note, state whether A2 genuinely continues A1,
adds something, repeats/paraphrases it, or feels disconnected. Report any
unclear words and whether artistic invention loses the connection to the image.
Do not infer a history-use result from these teacher clips.

### Verified preparation evidence

All 48 official teacher utterances complete with no frame caps, lasting
5.12–9.60 seconds each. Peak allocated VRAM is4.88GB. All 48 fragment strings
are distinct; the most repeated three-word opening occurs twice and no
three-word ending repeats. These lexical counts do not establish originality
or artistic merit. The eight listening sequences cover four source images
and three historical scene categories, with four sequences from each internal
partition. This subset is not a comprehensive validation of all ten categories.

The separate media-only cache exercises48 normalized audio vectors and12
verified historical image vectors at768D. No caption, transcript or variable
text enters its encoder call. The cached histories are teacher histories,
not demonstrations of self-generated-history robustness.

Whisper Small's auxiliary word-edit ratio is1.06% across the48 utterances.
This is a recognizer diagnostic, not a human intelligibility score, and is never
used to select or replace a sequence. Single-item sampled synthesis of the
initial diagnostic phrase also reproduces its isolated control byte-for-byte.
The [public aggregate manifest](validation/2026-10-16-asfl-two-fragment-corpus.json)
contains verified counts and private-evidence hashes, not the stories or media.

Verification: host25 passed/27 optional ML skips; worker CPU77 passed/4 skips;
locked vision4 passed; actual CUDA80 passed/2 vision skips. All scripts execute
in their intended host-or-managed-worker roles. Human corpus validation remains
pending; no optimizer, reader training or model promotion occurs.

## Reproduction and privacy

Private root: `artifacts/asfl-two-fragment-20261016/`. `selection/selection.json`,
raw `drafts/`, explicit `editorial-review*.json`, `corpus*/corpus.json`, machine
audits, teacher manifests, media-only exports and the listening mapping remain
local. Executed-source snapshots preserve pre-refactor source hashes; logs
retain failed build and synthesis attempts. Public source contains generic
prompts/validators and aggregate evidence only, never the48 private fragments.
Official generation is pinned to Qwen revision
`5d83992436eae1d760afd27aff78a71d676296fc`; the feedback encoder remains
EmbeddingGemma2 revision`914f7f89142e33e77833254d9c9b90c3cef7303b`.

Representative exact teacher command after the bounded decoding control:

```bash
DOCKER_CONFIG="$PWD/.cache/docker-config" .cache/tools/docker compose --profile gpu run --rm \
  -e LD_REF_TEXT= -v "$PWD/src:/workspace/src:ro" -v "$PWD/scripts:/workspace/scripts:ro" \
  experiment-gpu python scripts/synthesize_temporal_teachers.py \
  --corpus artifacts/asfl-two-fragment-20261016/corpus-v2/corpus.json \
  --protocol configs/research/asfl-two-fragment.json \
  --output-dir artifacts/asfl-two-fragment-reproduction \
  --batch-size 1 --codec0-sampling --fixed-seed
```

Every stage requires a fresh directory or a strictly provenance-checked resume.
The text-free media cache uses only verified historical image vectors and the
generated audio waveforms; captions, stories and transcripts are absent from
its input schema. ASR, when run, is auxiliary and cannot select or filter the
teacher material. Human validation remains a separate gate. No reader training,
paid job, private upload or final-narrator promotion occurs in this stage.
