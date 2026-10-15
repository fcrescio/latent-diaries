"""Offline text-prototype diagnostic; never a speech-inference lookup."""
import argparse
import json
from pathlib import Path

import torch

from latent_diaries.training_text_features import FrozenTrainingTextEncoder
from latent_diaries.vision import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--design', type=Path, required=True)
    parser.add_argument('--canonical-texts', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if not Path('/.dockerenv').exists() or args.output_dir.exists():
        raise ValueError('Use managed worker and a fresh output directory')
    from safetensors.torch import save_file

    canonical = json.loads(args.canonical_texts.read_text())['texts']
    rows = json.loads(args.dataset.read_text())['rows']
    design = {r['id']: r for r in json.loads(args.design.read_text())['rows']}
    classes = sorted({design[r['id']]['scene'] for r in rows if r['split'] == 'train'})
    encoder = FrozenTrainingTextEncoder('914f7f89142e33e77833254d9c9b90c3cef7303b', 'cuda')
    vectors = encoder.encode([canonical[k] for k in classes])
    repeated = encoder.encode([canonical[classes[0]]])
    error = float((repeated - vectors[:1]).abs().max())
    if error > 1e-4:
        raise ValueError('Training-text batch/single parity failed')
    features = torch.tensor([r['features'] for r in rows])
    expected = torch.tensor([classes.index(design[r['id']]['scene']) for r in rows])
    predicted = (features @ vectors.T).argmax(-1)
    summary = {}
    for split in ['train', 'heldout']:
        mask = torch.tensor([r['split'] == split for r in rows])
        summary[split] = {'count': int(mask.sum()),
                          'accuracy': float((predicted[mask] == expected[mask]).float().mean())}
    args.output_dir.mkdir(parents=True)
    save_file({'embeddings': vectors}, str(args.output_dir / 'text-prototypes.safetensors'))
    result = {'scope': 'Training-only crossmodal diagnostic against existing VLM labels',
              'summary': summary, 'batch_single_max_abs_error': error,
              'classes': classes, 'dataset_sha256': sha256(args.dataset),
              'source_sha256': sha256(Path(__file__)), 'text_encoder_source_sha256':
              sha256(Path('src/latent_diaries/training_text_features.py'))}
    (args.output_dir / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'summary': summary, 'batch_single_max_abs_error': error}), flush=True)


if __name__ == '__main__':
    main()
