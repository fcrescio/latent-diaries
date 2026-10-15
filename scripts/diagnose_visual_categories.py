"""Probe existing image vectors; all labels/predictions remain local."""
import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from latent_diaries.inversion import digest
from latent_diaries.visual_diagnostic import classification_metrics, ridge_scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--design', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not Path('/.dockerenv').exists() or args.output.exists():
        raise ValueError('Use managed worker and a new output')
    rows = json.loads(args.dataset.read_text())['rows']
    design = {r['id']: r for r in json.loads(args.design.read_text())['rows']}
    if any(r['split'] not in {'train', 'heldout'} for r in rows):
        raise ValueError('Training/development only')
    train = torch.tensor([r['split'] == 'train' for r in rows])
    if {r['scene_group'] for r in rows if r['split'] == 'train'} & {
            r['scene_group'] for r in rows if r['split'] == 'heldout'}:
        raise ValueError('Scene-group leakage')
    classes = sorted({design[r['id']]['scene'] for r in rows if r['split'] == 'train'})
    labels = torch.tensor([classes.index(design[r['id']]['scene']) for r in rows])
    features = torch.tensor([r['features'] for r in rows])
    permutation = torch.randperm(int(train.sum()), generator=torch.Generator().manual_seed(42))
    centroids = F.normalize(torch.stack([features[train][labels[train] == k].mean(0)
                                        for k in range(len(classes))]), dim=-1)
    scores = {'ridge': ridge_scores(features[train], labels[train], features, len(classes)),
              'permuted_ridge': ridge_scores(features[train], labels[train][permutation], features, len(classes)),
              'cosine_centroid': features @ centroids.T}
    majority = int(labels[train].bincount().argmax())
    scores['majority'] = F.one_hot(torch.full_like(labels, majority), len(classes))
    summary = {name: {split: classification_metrics(value[mask], labels[mask])
                     for split, mask in [('train', train), ('development', ~train)]}
               for name, value in scores.items()}
    result = {'scope': 'Development visual separability only; no speech acceptance', 'summary': summary,
              'alpha': 1.0, 'classes': classes, 'dataset_sha256': digest(args.dataset),
              'design_sha256': digest(args.design), 'source_sha256': digest(Path(__file__)),
              'predictions': {name: value.argmax(-1).tolist() for name, value in scores.items()},
              'ids': [r['id'] for r in rows]}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    args.output.chmod(0o600)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
