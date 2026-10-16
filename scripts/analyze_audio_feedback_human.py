"""Summarize real blinded ratings; private mappings and notes never enter public output."""
import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paired_summary(values):
    if not values:
        raise ValueError('Paired observations required')
    rng = random.Random(20261016)
    samples = sorted(mean(rng.choices(values, k=len(values))) for _ in range(20000))
    return {'images': len(values), 'mean_difference': mean(values), 'median_difference': median(values),
            'wins': sum(v > 0 for v in values), 'ties': sum(v == 0 for v in values),
            'losses': sum(v < 0 for v in values),
            'descriptive_image_bootstrap95': [samples[499], samples[19499]],
            'bootstrap_replicates': 20000, 'seed': 20261016,
            'statistical_significance_claimed': False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ratings', type=Path, required=True)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('Fresh outcome report required')
    exported = json.loads(args.ratings.read_text())
    plan = json.loads(args.protocol.read_text())
    expected = set(plan['development_ids'])
    if len(exported) != 28 or len({r['mapping']['clip_id'] for r in exported}) != 28:
        raise ValueError('All28 prespecified blind clips required')
    reviewers = {r['ratings'][0]['reviewer'] for r in exported if len(r['ratings']) == 1}
    if len(reviewers) != 1 or any(len(r['ratings']) != 1 for r in exported):
        raise ValueError('This report requires one common reviewer and one real rating per clip')
    by_image, by_method, duplicate = defaultdict(dict), defaultdict(list), defaultdict(list)
    fields = ['grounding', 'voice', 'evocation', 'surprise']
    for entry in exported:
        mapping, rating = entry['mapping'], entry['ratings'][0]
        if rating['reviewer'] not in reviewers or rating['clip_id'] != mapping['clip_id']:
            raise ValueError('Rating identity mismatch')
        if mapping['image_id'] not in expected or mapping['method'] in by_image[mapping['image_id']]:
            raise ValueError('Unexpected or duplicate condition')
        if any(type(rating[k]) is not int or not 1 <= rating[k] <= 5 for k in fields):
            raise ValueError('Real bounded ordinal human scores required')
        by_image[mapping['image_id']][mapping['method']] = rating
        by_method[mapping['method']].append(rating)
        duplicate[mapping['image_id'], mapping['audio_sha256']].append(rating)
    if set(by_image) != expected:
        raise ValueError('Incomplete image comparison')
    conditions = {}
    for method, ratings in by_method.items():
        required = 4 if method == 'shuffled_image_scorer' else 8
        if len(ratings) != required:
            raise ValueError('Prespecified method count differs')
        conditions[method] = {'rated_pairs': len(ratings),
            'mean_scores': {k: mean(r[k] for r in ratings) for k in fields},
            'grounding_histogram': dict(sorted(Counter(r['grounding'] for r in ratings).items())),
            'explicit_out_of_context_notes': sum('not in context' in r['note'].casefold() for r in ratings),
            'note_absence_is_not_positive_grounding': True}
    paired = {}
    for baseline in ['first_candidate', 'fixed_random', 'shuffled_image_scorer']:
        pairs = [row for row in by_image.values() if baseline in row]
        paired[baseline] = {field: paired_summary([
            row['image_audio_max'][field]-row[baseline][field] for row in pairs]) for field in fields}
    cohorts = {}
    for name, ids in [('original_four', plan['development_ids'][:4]), ('additional_four', plan['development_ids'][4:])]:
        cohorts[name] = paired_summary([by_image[k]['image_audio_max']['grounding']-
                                       by_image[k]['first_candidate']['grounding'] for k in ids])
    repeats = [rows for rows in duplicate.values() if len(rows) > 1]
    report = {'scope': 'Completed human no-training pilot; descriptive ordinal scores, no generalization claim',
        'source_sha256': sha(Path(__file__)), 'private_ratings_sha256': sha(args.ratings),
        'protocol_sha256': sha(args.protocol), 'rated_pairs': 28, 'source_images': 8, 'reviewers': 1,
        'conditions': conditions, 'image_audio_max_minus_baseline': paired,
        'cohort_grounding_gain_over_first': cohorts,
        'identical_media_repeat_groups': len(repeats),
        'repeat_groups_with_any_score_disagreement': sum(any(len({r[k] for r in rows}) > 1 for k in fields) for rows in repeats),
        'identical_media_are_not_independent_generations': True,
        'qualitative_user_feedback': 'Little overall change; familiar phrases with rare exceptions and persistent out-of-context cases.',
        'decision': 'Do not promote reranking as the narrator improvement; retain encoder feasibility as separate evidence.',
        'next_hypothesis': 'Test genuinely progressive, image-specific two-fragment supervision before temporal training; audio feedback alone cannot supply missing repertoire.',
        'temporal_training_started': False, 'reranker_promoted': False, 'm2_accepted': False,
        'public_output_has_no_private_notes_media_or_mapping': True}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'conditions': conditions, 'cohort_grounding_gain_over_first': cohorts}, indent=2))


if __name__ == '__main__':
    main()
