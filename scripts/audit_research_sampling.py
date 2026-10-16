"""Quantify actual V23 and V24 sampling from existing private logs/corpus."""
import argparse
import json
from collections import Counter
from pathlib import Path

from latent_diaries.feature_provenance import checksum


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repair-root', type=Path, required=True)
    p.add_argument('--corpus', type=Path, required=True)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('Do not overwrite historical audits')
    result = {'scope': 'Observed sampling logs and analytical sampling probabilities; no retraining', 'v23': []}
    training_ids = {r['id'] for r in json.loads(args.dataset.read_text())['rows'] if r['split'] == 'train'}
    for iteration in [0, 1]:
        manifest = args.repair_root / f'histories-{iteration}/manifest.json'
        rollouts = json.loads(manifest.read_text())['records']
        ids = {r['id'] for r in rollouts}
        path = args.repair_root / f'reader-{iteration}/loss.jsonl'
        log = [json.loads(line) for line in path.read_text().splitlines()]
        sources = Counter(r['history_source'] for r in log)
        if not {r['training_id'] for r in log} <= training_ids:
            raise ValueError('Sampling log contains non-training IDs')
        counts = Counter(dict.fromkeys(training_ids, 0));counts.update(r['training_id'] for r in log)
        selected = [counts[k] for k in ids]
        other = [v for k, v in counts.items() if k not in ids]
        result['v23'].append({'iteration': iteration, 'steps': len(log), 'source_counts': sources,
            'rollout_images': len(ids), 'distinct_training_images_observed': sum(v > 0 for v in counts.values()),
            'training_images_never_sampled': sum(v == 0 for v in counts.values()),
            'rollout_member_mean_exposures': sum(selected)/len(selected),
            'other_image_mean_exposures': sum(other)/len(other),
            'expected_per_rollout_member_probability': 0.5/len(training_ids) + 0.5/len(ids),
            'expected_per_other_probability': 0.5/len(training_ids),
            'dataset_sha256': checksum(args.dataset),
            'counts_by_training_id': counts, 'loss_log_sha256': checksum(path),
            'history_manifest_sha256': checksum(manifest),
            'cache_source_reader_sha256': json.loads(manifest.read_text())['source_reader_sha256']})
    rows = [r for r in json.loads(args.corpus.read_text())['rows'] if r['split'] == 'train']
    categories = Counter(r['category'] for r in rows)
    image_count = sum(v for k, v in categories.items() if k.startswith('image_'))
    result['v24'] = {'corpus_sha256': checksum(args.corpus), 'training_rows': len(rows),
        'image_specific_training_rows': image_count, 'replay_training_rows': len(rows)-image_count,
        'image_specific_sampling_probability': image_count/len(rows),
        'category_row_counts': categories, 'independent_image_groups': sum(k.startswith('image_') for k in categories),
        'interpretation': 'Uniform utterance sampling, not uniform images; language paraphrases do not add independent visual contexts'}
    args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
