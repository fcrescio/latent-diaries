"""Select existing private training images before stories, teacher synthesis or scoring."""
import argparse
import json
from pathlib import Path

from latent_diaries.feature_provenance import checksum
from latent_diaries.temporal_corpus import select_pilot


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['dataset', 'design', 'labels', 'protocol', 'output-dir']:
        p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args()
    if args.output_dir.exists():
        raise ValueError('Fresh private selection directory required')
    dataset = json.loads(args.dataset.read_text());design = json.loads(args.design.read_text())
    if dataset['design_sha256'] != checksum(args.design) or dataset['labels_sha256'] != checksum(args.labels):
        raise ValueError('Historical dataset provenance differs')
    labels = {r['id']: r for r in map(json.loads, args.labels.read_text().splitlines())}
    rows = select_pilot(dataset['rows'], {r['id']: r for r in design['rows']})
    for row in rows:
        label = labels[row['id']]
        if label.get('training_only') is not True or label['label'].get('keep') is not True:
            raise ValueError('Accepted original training inventory only')
        row['image_sha256'] = checksum(row['image'])
    review = []
    for split in ['train', 'validation']:
        for row in [r for r in rows if r['split'] == split][:2]:
            review.extend(row['id']+'-'+branch for branch in ['A', 'B'])
    result = {'scope': 'Private two-fragment corpus preparation; source images are original training only',
        'dataset_sha256': checksum(args.dataset), 'design_sha256': checksum(args.design),
        'labels_sha256': checksum(args.labels), 'protocol_sha256': checksum(args.protocol),
        'source_sha256': checksum(__file__), 'seed': 20261016, 'rows': rows,
        'review_sequence_ids': review, 'human_validated': False, 'neural_training': False}
    args.output_dir.mkdir(mode=0o700, parents=True)
    path = args.output_dir/'selection.json';path.write_text(json.dumps(result, indent=2)+'\n');path.chmod(0o600)
    print(json.dumps({'images': len(rows), 'train_images': 8, 'validation_images': 4,
                      'capture_groups': len({r['scene_group'] for r in rows}), 'review_sequences': len(review)}))


if __name__ == '__main__':
    main()
