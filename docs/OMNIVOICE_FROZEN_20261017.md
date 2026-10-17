# OMX-01 — frozen OmniVoice feasibility

Executed and reviewed 2026-10-17: **20 actual WAVs, technical checks passed,
human gate failed — NO-GO for this frozen textless-target hypothesis**. This is an isolated comparator. Qwen remains the main backbone; no
reader, adapter, LoRA or training was run. M1/M2 are not claimed for OmniVoice.
Machine-readable facts: [validation](validation/omnivoice-frozen-20261017.json).

The user supplied `/tmp/omnivoice_probe/` and explicitly requested execution.
A private exact copy of the protocol and original harness is retained with
hashes in the ignored experiment directory. Its bounded exception permits a
consented, immutable voice reference **and its exact transcript**. No variable
target text is allowed in textless-target arms. General AGENTS instructions
are unchanged. The reference transcript came from the already authorized local
configuration, not ASR; it and the WAV remain private.

Source: [official pinned code](https://github.com/k2-fsa/OmniVoice/tree/08be0b4ccbac3e13e374e86fbfead4b4cac343e2).
Checkpoint: [fixed snapshot](https://huggingface.co/k2-fsa/OmniVoice/tree/c5fdb5ccb189668d56333f77ba2629f4cd7535f4).
All 13 snapshot files were size/hash checked against Hub metadata where LFS
SHA256 was available. Inference had no network, no images, no ASR, readonly
snapshot/reference mounts and frozen weights. Model code matched the installed
pinned checkout. Primary Qwen and vision locks were not changed.

## Prespecified experiment

| Arm | Conditioning | Cases |
| --- | --- | ---: |
| Positive | Official TTS API, two fixed Italian target sentences, fixed voice | 2 |
| Voice-empty | Empty target, Italian marker, fixed WAV + exact reference transcript | 6 |
| Lang-empty | Empty target, Italian marker, no speaker reference | 6 |
| Mask-only | Pure `[1,8,F]` acoustic MASK IDs; no text, language or voice | 6 |

Textless-target does **not** mean absence of all textual vectors in voice-empty
or lang-empty: upstream inserts markers, language and optionally reference text.
Only mask-only removes every text position. It uses the private iterative API;
both CFG branches were checked identical on all 32 steps. Positive arms retain
the official preparation/generation/decode path, with pass-through audits.

Seeds 41/42/43, durations 4/6 seconds, 32 iterations, CFG 2, position temperature
5, class temperature 0 and `postprocess_output=False` were fixed before running.
The four smoke outputs were reused only after provenance, code and WAV hash
checks; the full stage generated the remaining sixteen. All twenty WAVs have
distinct hashes. This does not prove novel or intelligible content.

Every decoded code was an integer in `[0,1023]`, with no MASK remaining. All
waveforms were finite, nonempty and different from the reference file. None
crossed the RMS silence-candidate threshold or clipped. Actual lengths are
4.2/6.2 seconds because official processing retains its default fade/padding.
Peak allocated VRAM was 2,126,612,992 bytes. Summed generation calls took about
20.46 seconds, excluding loading, fixed prompt preparation and file writing.

## Reproduction

The executed worker image was
`sha256:ef437169811f3fb35c3e31a66119a5115009b940adb73b3ac3b0d07f9fc49fd6`,
with readonly source overlays from revision `2308535`. The isolated interpreter
and full dependency lock are under `environments/omnivoice/`; no weights are
baked into the image. The CUDA stack was actually imported and exercised,
including the bundled codec. See that directory's README for the binary pair.

The snapshot cache lives on the project filesystem because the Docker internal
disk had under 2 GB free. No existing caches were pruned. The override replaces
inherited mounts so private images and historical model caches are unavailable.
The actual smoke command was:

```bash
DOCKER_CONFIG="$PWD/.cache/docker-config" .cache/tools/docker compose \
  -f docker-compose.yml -f docker/compose.omnivoice.yml --profile gpu \
  run --rm experiment-gpu /opt/omnivoice/.venv/bin/python \
  scripts/omnivoice_frozen_probe.py --phase smoke \
  --model-snapshot /cache/omnivoice/hub/models--k2-fsa--OmniVoice/snapshots/c5fdb5ccb189668d56333f77ba2629f4cd7535f4 \
  --reference-wav /workspace/run-data/voices/reference.wav \
  --reference-text-file /workspace/run-data/voices/reference.txt \
  --source-commit 08be0b4ccbac3e13e374e86fbfead4b4cac343e2 \
  --model-revision c5fdb5ccb189668d56333f77ba2629f4cd7535f4 \
  --output-dir /workspace/artifacts/omnivoice-feasibility-20261017/outputs \
  --device cuda:0 --dtype float16
```

The full command was identical except `--phase full --skip-existing`. Never
rerun over a partial case, tune seeds after listening, or silently substitute
API/model versions. Runtime/source/config fingerprints must match on reuse.
Before GPU use: isolated `py_compile` and `--help`, pure protocol tests, actual
API signatures and a tiny frozen real iterative-sampler CPU test passed. The
initial tiny fixture's text vocabulary was too small for MASK ID 1024; fixing
the fixture to 2048 made the four tests pass. No upstream modification or
experimental retry was needed. Logs are retained privately.

## Local human review

The authenticated gallery at `/audio-probe` has twenty hash-ordered labels
**OMX-01 … OMX-20** and no images. The private mapping is outside Git. Existing
image-pair scores are preserved; OMX uses separate tables and scales.

For every generated clip, record language, intelligibility 0–4, novelty versus
the original reference, optional coherence 0–4 and a note. Zero is a real score,
missing values stay missing. Reference audio is clearly separate from generated
audio. No field is prefilled. Voice similarity 1–5 appears only after all twenty
core votes, and only for the eight reference-conditioned clips, avoiding an
applicability clue during blinded listening. Basic auth, CSRF and path checks
also cover these routes. No tunnel, public media upload or new host exposure
was introduced; the previously authorized local bind remains in use.

Ingest, after the verified complete grid, was through the same managed worker:

```bash
DOCKER_CONFIG="$PWD/.cache/docker-config" .cache/tools/docker compose \
  --profile experiment run --rm \
  -v "$PWD/src:/workspace/src:ro" -v "$PWD/scripts:/workspace/scripts:ro" \
  -v "$PWD/services:/workspace/services:ro" experiment \
  python -m scripts.prepare_omnivoice_review \
  --outputs artifacts/omnivoice-feasibility-20261017/outputs \
  --reference run-data/voices/reference.wav \
  --mapping artifacts/omnivoice-feasibility-20261017/blind-mapping.json
```

At preparation there were zero judgments; all twenty core judgments are now
recorded. Both official positive controls are rated fluent Italian (4/4). ASR was not used and cannot replace
human listening. Baseline unintelligibility requires stopping quality comparison.
H1 passes only with at least 4/6 fluent/coherent Italian scores of 4, at least one
success per duration, and at least two distinct novel contents beyond reference.
H2/H3 are informative controls. A passing gate permits only a proposal for a
numeric semantic-oracle test, not adapter training. Otherwise stop this frozen
hypothesis without scaling or sampling searches. Operational errors would be
indeterminate, not evidence of bad model quality.

## Licenses and publication

Repository code is [Apache 2.0](https://github.com/k2-fsa/OmniVoice/blob/08be0b4ccbac3e13e374e86fbfead4b4cac343e2/LICENSE).
The pinned main [model card](https://huggingface.co/k2-fsa/OmniVoice/blob/c5fdb5ccb189668d56333f77ba2629f4cd7535f4/README.md)
declares CC-BY-NC without identifying a version. The bundled acoustic tokenizer
has a separate [Boson Higgs Audio 2 Community License](https://huggingface.co/k2-fsa/OmniVoice/blob/c5fdb5ccb189668d56333f77ba2629f4cd7535f4/audio_tokenizer/LICENSE),
including restrictions concerning other-model improvement and competitive use.
These are separate evidence, not a claim that Apache covers all weights or
that commercial redistribution is cleared. This run is local research, with
no vendor-weight redistribution and no training on generated outputs.

Only generic source, dependency locks and aggregate facts are published. WAVs,
reference transcript, codes, private SHA256s, per-case mapping and comments stay
in ignored local storage. The review assets can be deleted locally with the
corresponding database entries when no longer needed; no remote media copy was
made.


## Completed human outcome — NO-GO

All 20 core votes from one reviewer were read from the local database and
snapshotted privately. No optional voice-similarity or coherence scores were
submitted; these remain missing and are not needed to resolve the failed gate.

| Arm | Human intelligibility | Novelty judgments |
| --- | --- | --- |
| Positive (2) | Both 4/4, Italian | Undecidable (not an H1 novelty test) |
| Voice-empty (6) | One 0, one 2, four 3; none 4 | Four repeats, two undecidable; none new |
| Lang-empty (6) | All 0: silence/nonvoice | All undecidable |
| Mask-only (6) | All 0: silence/nonvoice | All undecidable |

Overall 13/20 have score 0, five are partially intelligible and only the two
text-conditioned controls are fluent Italian. The four score-3 voice-empty
outputs are marked mixed-language repetitions. H1 has 0/6 successes against
its prespecified requirement of at least 4/6 and two distinct novel contents.
Both duration strata have zero H1 successes. H2/H3 provide no positive control
for spontaneous speech. The result is a model-quality NO-GO for this exact
frozen empty-target setup, not an operationally indeterminate run.

The successful positive controls show that this installation can produce
Italian when given target text. They do not prove that a particular internal
mechanism caused the empty-target failure. Above-threshold RMS merely measured
nonzero energy: human silence/noise judgments supersede any suggestion that it
implied speech. This is not evidence that OmniVoice cannot be retrained, nor a
comparative TTS ranking against Qwen.

Stop here as the supplied protocol requires: no seed/CFG/temperature search,
no adapter or image-reader training, no semantic-oracle progression. Preserve
all evidence. Further architectural alternatives require a separate proposal
and their own competence gate; this experiment does not justify scaling.
