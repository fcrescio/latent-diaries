"""Training-only numeric handoff from pinned semantic text encoder to Qwen reader."""
import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from latent_diaries.training_text_features import FrozenTrainingTextEncoder
from latent_diaries.vision import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if not Path('/.dockerenv').exists() or args.output_dir.exists():
        raise ValueError('Use managed worker and fresh cache')
    rows = json.loads(args.corpus.read_text())['rows']
    encoder = FrozenTrainingTextEncoder('914f7f89142e33e77833254d9c9b90c3cef7303b', 'cuda')
    values = []
    for i in range(0, len(rows), 16):
        values.append(encoder.encode([r['text'] for r in rows[i:i + 16]]))
        if i % 160 == 0:
            print(f'Encoded training text {min(i + 16, len(rows))}/{len(rows)}', flush=True)
    args.output_dir.mkdir(parents=True)
    save_file({'embeddings': torch.cat(values)}, str(args.output_dir / 'features.safetensors'))
    manifest = {'corpus_sha256': sha256(args.corpus), 'source_sha256': sha256(Path(__file__)),
                'encoder_source_sha256': sha256(Path('src/latent_diaries/training_text_features.py')),
                'features_sha256': sha256(args.output_dir / 'features.safetensors'),
                'vision_lock_sha256': sha256(Path('/opt/vision/uv.lock')),
                'model_revision': '914f7f89142e33e77833254d9c9b90c3cef7303b',
                'format': 'title: none | text: {training sentence}', 'ids': [r['id'] for r in rows]}
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
