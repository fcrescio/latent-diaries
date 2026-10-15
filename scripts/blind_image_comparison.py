"""Import a bounded two-reader development comparison into the local gallery."""
import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageOps

from latent_diaries.review import ingest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--design', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--baseline-results', type=Path)
    parser.add_argument('--candidate-only', action='store_true', help='Use prior baseline reviews without repeating their clips')
    parser.add_argument('--acoustic-results', '--candidate-results', dest='candidate_results', type=Path, required=True)
    parser.add_argument('--candidate-name', default='acoustic')
    parser.add_argument('--include-shuffled', action='store_true')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--label-prefix', default='E5-')
    args = parser.parse_args()
    if args.candidate_only == bool(args.baseline_results):
        raise ValueError('Provide baseline results or choose candidate-only review')
    if args.output_dir.exists() or not all(c.isalnum() or c in '-_' for c in args.run_id):
        raise ValueError('Use fresh output directory and safe run ID')
    if not args.label_prefix or not all(c.isascii() and (c.isalnum() or c in '-_') for c in args.label_prefix):
        raise ValueError('Use a safe blind-label prefix')
    if args.candidate_name == 'baseline' or not args.candidate_name or not all(
            c.isalnum() or c in '-_' for c in args.candidate_name):
        raise ValueError('Use a distinct safe candidate method name')
    design = {r['id']: r for r in json.loads(args.design.read_text())['rows']}
    plan = json.loads(args.plan.read_text())
    ids = plan['heldout_ids'] + [plan['training_id']]
    if len(ids) != len(set(ids)) or any(design[k]['split'] != 'heldout' for k in plan['heldout_ids']):
        raise ValueError('Invalid development comparison plan')
    if design[plan['training_id']]['split'] != 'train':
        raise ValueError('Invalid training control')
    pairs = []
    comparisons = [] if args.candidate_only else [('baseline', args.baseline_results)]
    comparisons.append((args.candidate_name, args.candidate_results))
    for method, path in comparisons:
        results = json.loads(path.read_text())
        if results['text_used'] or results['teacher_files_read'] or results['linguistic_labels_read']:
            raise ValueError('Comparison must use standalone label-free inference')
        by_id = {r['id']: r for r in results['results']}
        if set(by_id) != set(ids + ['null']):
            raise ValueError('Generated IDs differ from frozen plan')
        for label in ids + ['null']:
            row = by_id[label]
            if not row.get('audio') or not Path(row['audio']).is_file():
                raise ValueError('Missing genuine generated audio')
            pairs.append({'method': method, 'audio_id': label,
                          'image_id': label if label != 'null' else ids[0],
                          'condition': 'null' if label == 'null' else ('train' if label == plan['training_id'] else 'matched'),
                          'result': row})
        if args.include_shuffled and method == args.candidate_name:
            order = plan['shuffle']
            held = plan['heldout_ids']
            if sorted(order) != list(range(len(held))) or any(i == j for i, j in enumerate(order)):
                raise ValueError('Shuffled control must be a complete derangement')
            for i, label in enumerate(held):
                audio_id = held[order[i]]
                pairs.append({'method': method, 'audio_id': audio_id, 'image_id': label,
                              'condition': 'shuffled', 'result': by_id[audio_id]})
    if len(pairs) > 26:
        raise ValueError('Too many clips for alphabetic labels')
    random.Random(2026101505).shuffle(pairs)
    args.output_dir.mkdir(parents=True, mode=0o700)
    images = {}
    for label in ids:
        destination = args.output_dir / f'{label}.jpg'
        with Image.open(design[label]['path']) as original:
            image = ImageOps.exif_transpose(original).convert('RGB')
            image.thumbnail((1024, 1024));image.save(destination, quality=90)
        images[label] = str(destination.resolve())
    rows, mapping = [], []
    for i, pair in enumerate(pairs):
        letter = chr(65 + i)
        row = {'id': f'{args.run_id}-{letter.lower()}', 'run_id': args.run_id,
               'blind_label': args.label_prefix + letter, 'image': images[pair['image_id']],
               'audio': str(Path(pair['result']['audio']).resolve())}
        rows.append(row)
        mapping.append(pair | {'blind_label': row['blind_label'], 'clip_id': row['id']})
    manifest = args.output_dir / 'review.jsonl'
    manifest.write_text(''.join(json.dumps(r) + '\n' for r in rows));manifest.chmod(0o600)
    private = args.output_dir / 'private-mapping.json'
    private.write_text(json.dumps(mapping, indent=2) + '\n');private.chmod(0o600)
    count = ingest(manifest, Path('run-data/reviews.sqlite3'), Path('run-data/review-assets'))
    print(f'Imported {count} genuine generated development pairs; no ratings created')


if __name__ == '__main__':
    main()
