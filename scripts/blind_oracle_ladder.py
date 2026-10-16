"""Import a small blinded DIAGNOSTIC listening set; never fabricate ratings."""
import argparse
import json
import random
from pathlib import Path

from latent_diaries.feature_provenance import checksum
from latent_diaries.review import ingest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--design', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--run-id', required=True)
    args = p.parse_args()
    if args.output_dir.exists() or not all(c.isascii() and (c.isalnum() or c in '-_') for c in args.run_id):
        raise ValueError('Fresh output directory and safe run ID required')
    results = json.loads(args.results.read_text())
    design = {r['id']: r for r in json.loads(args.design.read_text())['rows']}
    chosen = [r for r in results['results'] if r.get('level') in {'L0', 'L1', 'L2', 'L3', 'L4'} and
              (r['seed'] == 42 or (r['image_id'] == 'p0003' and r['level'] in {'L1', 'L2'}))]
    if len(chosen) != 14:
        raise ValueError('Expected fixed fourteen-clip two-image diagnostic selection')
    for r in chosen:
        if design[r['image_id']]['split'] != 'train' or checksum(Path(r['audio'])) != r['audio_sha256']:
            raise ValueError('Verified training-oracle diagnostic audio required')
    random.Random(2026101602).shuffle(chosen)
    args.output_dir.mkdir(mode=0o700)
    manifest = args.output_dir / 'review.jsonl'
    rows, mapping = [], []
    for i, record in enumerate(chosen):
        label = 'OD-' + chr(65 + i)
        row = {'id': args.run_id + '-' + chr(97+i), 'run_id': args.run_id, 'blind_label': label,
               'image': str(Path(design[record['image_id']]['path']).resolve()),
               'audio': str(Path(record['audio']).resolve())}
        rows.append(row);mapping.append({'blind_label': label, 'clip_id': row['id'], **record})
    manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows));manifest.chmod(0o600)
    private = args.output_dir / 'private-mapping.json'
    private.write_text(json.dumps({'scope': 'Oracle diagnostic; contains offline text controls, NOT a textless candidate gallery',
                                  'results_sha256': checksum(args.results), 'selection_seed': 2026101602,
                                  'mapping': mapping}, indent=2)+'\n');private.chmod(0o600)
    count = ingest(manifest, Path('run-data/reviews.sqlite3'), Path('run-data/review-assets'))
    print(f'Imported {count} real diagnostic clips; zero ratings created')


if __name__ == '__main__':
    main()
