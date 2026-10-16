# V25 proposal — on-policy stability of the existing continuous prior

Status: **PROPOSED ONLY, NOT AUTHORIZED OR STARTED.** Read
`REVIEW_RESPONSE_20261016.md` and its diagnostic listening results first.
The user review explicitly says: “Non eseguire V25 automaticamente. Presentare
prima i risultati diagnostici e una raccomandazione motivata.”

## One hypothesis

The first important failure is inside the continuous language prior: its
candidates can approximate pronounceable gold-history prefixes, but routing and
recurrent context drift when previous vectors are generated. Training part of
the SAME prior objective on fresh on-policy continuous histories may improve
L2/L3 stability without changing the acoustic backbone or visual features.
This is distinct from V23, which repairs Qwen audio policies using stale audio
histories. V25 would repair the earlier continuous-prefix generator instead.

The supplemental gold-history routed/best-candidate controls retain far more
recognizable Italian in auxiliary ASR than free-run L2. This is supporting
diagnostic evidence, not human approval. If listening does not confirm that
oracle best-candidate prefixes are reasonably intelligible, reject this proposal:
closed-loop training would be premature when candidate geometry itself fails.

## Fixed elements and isolated change

Keep V24 corpus/splits,768-dimensional subspace,32 modes, all normalization
buffers, reader width, semantic perturbations, image adapter, decoding and both
vendor backbones unchanged. No labels, images, LoRA, language vocabulary,
text-token projection/retrieval or learned backbone changes.

Two bounded continuation arms start from the identical existing V24 checkpoint,
with the same reset optimizer/schedule and prespecified minibatch/semantic draws:

- Control: existing gold-previous-vector objective.
- Treatment:80% identical gold-history objective and20% fresh on-policy
  continuous-prefix history objective, with the same reconstruction/routing/stop
  terms and weights. Target vectors remain training supervision only.

Generate histories from the current student at each update; no offline history
bank. Use bounded truncated gradients and an independent sampling RNG so changing
history generation does not silently change training minibatches or augmentation.
Never replace inference histories with gold vectors. Save checkpoint/RNG/optimizer
state, actual resume200→400 steps per arm, and exact changed-parameter provenance.
No further step, width, mode, alpha or LoRA sweep is part of this proposal.

## Benchmark and stop rules

Before fitting, freeze four additional already annotated training-image groups,
selected without looking at outputs, plus the two diagnostic groups. Use exact
training semantics L2 and oracle means L3 to isolate the prior; language-held-out
paraphrases remain within training-image groups, not held-out images. Test
seeds41/42/43. Reuse the fixed acoustic decoder, speaker and160-frame cap.
Image-only development is secondary; leave all96 reserved images untouched.

Compare control/treatment free-run errors, routing entropy and agreement,
length/stop/cap behavior, and blinded listening. Lower teacher-forced MSE or
better ASR is insufficient. Stop if free-run stability does not improve over the
matched continuation control, intelligibility regresses, modes collapse, or
repetition/cap failures increase. Do not claim M2 without a later positive blind
matched/shuffled/null image-grounding evaluation.

## Proposed resource bound and authorization

Local RTX4090 only, same managed worker image, no HF job, spending flag change,
private upload or additional backbone download. Propose400 updates per arm,
checkpoint resume at200, inference in small batches, and an initial20-minute GPU
wall-clock ceiling. This is an estimate to validate with a short memory/timing
probe, not a measured run. Cap intended peak allocation at4GiB; stop rather than
silently enlarge the budget. Local resources imply no new cloud invoice; this
is not a claim that local energy/time are free.

First obtain the focused human diagnostic assessment and explicit instruction
to execute V25. Until then only the diagnostics and confirmed bug fixes are
operational. The existing Italian-stable reference remains preserved.
