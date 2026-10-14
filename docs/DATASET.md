# Dataset and curation contract

JSONL each row:

```json
{"id":"walk01-frame001","scene_group":"walk01","image":"images/001.jpg","audio":"audio/001.wav","text_train_only":"Un giorno la città...","language":"Italian","speaker_id":"narrator_01","style":"intimate","consent":"self-recorded"}
```

Paths must resolve under a private local dataset root. Reference voices and sensitive wearable images should not be shipped in public datasets. Use separate metadata to document rights to image, speaker/audio, and text separately. Prefer one consented narrator; avoid voice imitation without permission. Make clips ~6–12 seconds at 24kHz for starting experiments. Split on `scene_group`, preventing adjacent frames from straddling train/validation/test. For voice safety, avoid writing identification labels of people in the image into narrations without consent.

Two stages: grounded spoken descriptions, then curator-authored non-literal narrative spoken by a consented voice. Text can be used for teacher alignment in training, never consumed in final inference. Keep at least one visually contrasted paired evaluation set, and null/shuffle controls. Multiple valid narratives/image allowed only once grounding is demonstrated. Audio codecs can be cached; raw user audio remains outside git/HF public.

`run-data/manifests/review-v1.jsonl` per line:

```json
{"id":"run1-0001","run_id":"run1","blind_label":"condition-unknown-to-reviewer","image":"../images/0001.jpg","audio":"../outputs/0001.wav"}
```

The evaluation importer resolves paths relative to the manifest. It copies allowed local media into a private review directory, generates hashed filenames, and only exposes them to an authenticated user on a local-only server. **Do not treat CSV scores as public by default.**
