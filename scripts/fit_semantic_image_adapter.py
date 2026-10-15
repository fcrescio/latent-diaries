"""Train-only numeric modality calibration; runtime loads a coefficient tensor only."""
import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from latent_diaries.inversion import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image-dataset', type=Path, required=True)
    parser.add_argument('--design', type=Path, required=True)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if not Path('/.dockerenv').exists() or args.output_dir.exists():
        raise ValueError('Use managed worker and a new adapter directory')
    rows = json.loads(args.image_dataset.read_text())['rows']
    design = {r['id']: r for r in json.loads(args.design.read_text())['rows']}
    text = json.loads(args.corpus.read_text())['rows']
    vectors = load_file(str(args.features))['embeddings']
    categories = sorted({r['category'] for r in text if r['split'] == 'train'})
    means = {k: torch.nn.functional.normalize(torch.stack([vectors[i] for i, r in enumerate(text)
             if r['split'] == 'train' and r['category'] == k]).mean(0), dim=0) for k in categories}
    training = [r for r in rows if r['split'] == 'train' and design[r['id']]['scene'] in means]
    x = torch.tensor([r['features'] for r in training], dtype=torch.float64)
    y = torch.stack([means[design[r['id']]['scene']] for r in training]).double()
    center_x, center_y = x.mean(0), y.mean(0)
    centered = x - center_x
    weights = centered.T @ torch.linalg.solve(centered @ centered.T + torch.eye(len(x)), y - center_y)
    coefficient = torch.cat([(center_y - center_x @ weights)[None], weights], 0).float()
    if not torch.isfinite(coefficient).all():
        raise RuntimeError('Invalid semantic calibration')
    args.output_dir.mkdir(parents=True)
    save_file({'coefficient': coefficient}, str(args.output_dir / 'adapter.safetensors'))
    info = {'scope': 'Existing training images and training-text means only', 'alpha': 1.0,
            'training_cases': len(training), 'training_cases_excluded': sum(r['split'] == 'train' for r in rows) - len(training),
            'supported_categories': sorted({design[r['id']]['scene'] for r in training}),
            'image_dataset_sha256': digest(args.image_dataset), 'design_sha256': digest(args.design),
            'corpus_sha256': digest(args.corpus), 'features_sha256': digest(args.features),
            'source_sha256': digest(Path(__file__)),
            'protocol_sha256': digest(Path('configs/research/semantic-image-adapter-v6.json')),
            'adapter_sha256': digest(args.output_dir / 'adapter.safetensors')}
    (args.output_dir / 'manifest.json').write_text(json.dumps(info, indent=2) + '\n')
    print(json.dumps({'training_cases': len(training), 'supported_categories': len(info['supported_categories']),
                      'excluded_training_cases': info['training_cases_excluded']}), flush=True)


if __name__ == '__main__':
    main()
