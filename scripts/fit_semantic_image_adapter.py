"""Train-only numeric modality calibration; runtime loads a coefficient tensor only."""
import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from latent_diaries.inversion import digest
from latent_diaries.semantic_adapter import fit_coefficient, select_alpha
from latent_diaries.visual_anchors import training_anchor_targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image-dataset', type=Path, required=True)
    parser.add_argument('--design', type=Path, required=True)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--select-alpha', action='store_true', help='Train-group cross-validation only')
    parser.add_argument('--anchor-labels', type=Path, help='Training-only validated visual-anchor labels')
    parser.add_argument('--anchor-protocol', type=Path, default=Path('configs/research/visual-anchors-v10.json'))
    args = parser.parse_args()
    if not Path('/.dockerenv').exists() or args.output_dir.exists():
        raise ValueError('Use managed worker and a new adapter directory')
    if args.anchor_labels and not args.select_alpha:
        raise ValueError('Visual-anchor control requires training-group alpha selection')
    rows = json.loads(args.image_dataset.read_text())['rows']
    design = {r['id']: r for r in json.loads(args.design.read_text())['rows']}
    text = json.loads(args.corpus.read_text())['rows']
    vectors = load_file(str(args.features))['embeddings']
    keys = [f"{r['category']}:{r['opening_index']}" if args.anchor_labels else r['category'] for r in text]
    categories = sorted({keys[i] for i, r in enumerate(text) if r['split'] == 'train'})
    means = {k: torch.nn.functional.normalize(torch.stack([vectors[i] for i, r in enumerate(text)
             if r['split'] == 'train' and keys[i] == k]).mean(0), dim=0) for k in categories}
    targets = {r['id']: design[r['id']]['scene'] for r in rows if r['split'] == 'train'}
    if args.anchor_labels:
        labels = [json.loads(line) for line in args.anchor_labels.read_text().splitlines() if line]
        targets = training_anchor_targets(labels, set(targets))
    training = [r for r in rows if r['split'] == 'train' and targets.get(r['id']) in means]
    x = torch.tensor([r['features'] for r in training])
    y = torch.stack([means[targets[r['id']]] for r in training])
    alpha, validation = 1.0, None
    if args.select_alpha:
        protocol = json.loads(Path('configs/research/grounding-adapter-v9.json').read_text())
        alpha, validation = select_alpha(x, y, [r['scene_group'] for r in training],
                                         protocol['alpha_grid'], protocol['folds'], protocol['seed'])
    coefficient = fit_coefficient(x, y, alpha)
    if not torch.isfinite(coefficient).all():
        raise RuntimeError('Invalid semantic calibration')
    args.output_dir.mkdir(parents=True)
    save_file({'coefficient': coefficient}, str(args.output_dir / 'adapter.safetensors'))
    info = {'scope': 'Existing training images and training-text means only', 'alpha': alpha,
            'train_group_validation': validation,
            'training_cases': len(training), 'training_cases_excluded': sum(r['split'] == 'train' for r in rows) - len(training),
            'supported_categories': sorted({targets[r['id']] for r in training}),
            'anchor_labels_sha256': digest(args.anchor_labels) if args.anchor_labels else None,
            'image_dataset_sha256': digest(args.image_dataset), 'design_sha256': digest(args.design),
            'corpus_sha256': digest(args.corpus), 'features_sha256': digest(args.features),
            'source_sha256': digest(Path(__file__)),
            'protocol_sha256': digest(args.anchor_protocol if args.anchor_labels else
                                     Path('configs/research/grounding-adapter-v9.json' if args.select_alpha
                                          else 'configs/research/semantic-image-adapter-v6.json')),
            'calibration_source_sha256': digest(Path('src/latent_diaries/semantic_adapter.py')),
            'adapter_sha256': digest(args.output_dir / 'adapter.safetensors')}
    (args.output_dir / 'manifest.json').write_text(json.dumps(info, indent=2) + '\n')
    print(json.dumps({'alpha': alpha, 'train_group_validation': validation,
                      'training_cases': len(training), 'supported_categories': len(info['supported_categories']),
                      'excluded_training_cases': info['training_cases_excluded']}), flush=True)


if __name__ == '__main__':
    main()
