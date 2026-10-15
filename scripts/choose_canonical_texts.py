"""Choose equal-Qwen-prefix-length training sentences without seeing images."""
import argparse
import json
from pathlib import Path

import qwen_tts  # noqa: F401 -- registers the official processor
from huggingface_hub import snapshot_download
from qwen_tts import Qwen3TTSModel
from transformers import AutoProcessor

from latent_diaries.canonical_supervision import PHRASES, canonical_candidates
from latent_diaries.inversion import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not Path('/.dockerenv').exists() or args.output.exists():
        raise ValueError('Use managed worker and a new output')
    info = json.loads(Path('artifacts/m0_italian.json').read_text())
    snapshot = snapshot_download(info['model_id'], revision=info['model_revision'], local_files_only=True)
    processor = AutoProcessor.from_pretrained(snapshot)
    choices = {}
    for scene in PHRASES:
        by_length = {}
        for text in canonical_candidates(scene):
            # The pinned official helper uses no model state; processor is CPU-only.
            wrapped = Qwen3TTSModel._build_assistant_text(None, text)
            ids = processor(text=wrapped, return_tensors='pt', padding=True)['input_ids']
            if ids.ndim == 1:
                ids = ids[None]
            length = ids[:, 3:-5].shape[1]
            by_length.setdefault(length, text)
        choices[scene] = by_length
    common = set.intersection(*(set(x) for x in choices.values()))
    if not common:
        raise ValueError('No shared prefix length; revise authored candidates before fitting')
    # Fixed deterministic selection: shortest common length, first authored match.
    length = min(common)
    result = {'prefix_length': length, 'texts': {k:v[length] for k,v in choices.items()},
              'model_id': info['model_id'], 'model_revision': info['model_revision'],
              'worker_lock_sha256': digest(Path('uv.lock')),
              'supervision_source_sha256': digest(Path('src/latent_diaries/canonical_supervision.py')),
              'selection': 'shortest common token length; independent of all image labels'}
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'scene_count':len(choices),'constant_prefix_length':length}),flush=True)


if __name__ == '__main__':
    main()
