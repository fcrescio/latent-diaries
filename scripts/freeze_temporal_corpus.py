"""Freeze explicitly edited private stories; never equate editorial checks with human approval."""
import argparse
import json
from pathlib import Path

from latent_diaries.feature_provenance import checksum
from latent_diaries.temporal_corpus import validate_story


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['selection', 'drafts-dir', 'editorial-review', 'output-dir']:
        p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args()
    if args.output_dir.exists():
        raise ValueError('Fresh sealed corpus directory required')
    selected = json.loads(args.selection.read_text())
    edits = json.loads(args.editorial_review.read_text())
    if edits['human_validated'] is not False or edits['selection_sha256'] != checksum(args.selection):
        raise ValueError('Explicit non-human editorial provenance required')
    by_id = {r['id']: r for r in edits['records']}
    if len(by_id) != 12 or set(by_id) != {r['id'] for r in selected['rows']}:
        raise ValueError('All12 fixed image identities must be reviewed, never silently replaced')
    rows = []
    for image in selected['rows']:
        edit = by_id[image['id']]
        raw = args.drafts_dir/(image['id']+'.raw.json')
        if edit['original_draft_sha256'] != checksum(raw) or not edit['editorial_notes']:
            raise ValueError('Exact source draft and explicit editorial rationale required')
        story = validate_story(edit['story'])
        if checksum(image['image']) != image['image_sha256']:
            raise ValueError('Selected pixels changed')
        for branch in story['branches']:
            rows.append({'id': image['id']+'-'+branch['key'], 'image_id': image['id'],
                'split': image['split'], 'scene_group': image['scene_group'], 'scene': image['scene'],
                'image': image['image'], 'image_sha256': image['image_sha256'], 'features': image['features'],
                'visible_details': story['visible_details'], 'fragments': branch['fragments'],
                'focus_it': branch['focus_it'], 'progression_it': branch['progression_it'],
                'a2_requires_a1_it': branch['a2_requires_a1_it'], 'training_only_text': True,
                'original_draft_sha256': checksum(raw), 'human_validated': False})
    if len(rows) != 24 or not set(selected['review_sequence_ids']) <= {r['id'] for r in rows}:
        raise ValueError('Complete24-sequence corpus and frozen review identities required')
    partitions = [{r['scene_group'] for r in rows if r['split'] == split} for split in ['train', 'validation']]
    if partitions[0] & partitions[1]:
        raise ValueError('Capture-group split leakage')
    args.output_dir.mkdir(mode=0o700)
    result = {'scope': 'Training-only teacher-story upper bound, not textless generated narration',
        'source_sha256': checksum(__file__), 'selection_sha256': checksum(args.selection),
        'editorial_review_sha256': checksum(args.editorial_review),
        'drafting_manifest_sha256': checksum(args.drafts_dir/'manifest.json'),
        'protocol_sha256': selected['protocol_sha256'], 'rows': rows,
        'review_sequence_ids': selected['review_sequence_ids'],
        'human_subset_validation_pending': True, 'neural_training_permitted': False}
    path = args.output_dir/'corpus.json';path.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n');path.chmod(0o600)
    print(json.dumps({'sequences': len(rows), 'utterances': sum(len(r['fragments']) for r in rows),
                      'human_validation_pending': True, 'training_started': False}))


if __name__ == '__main__':
    main()
