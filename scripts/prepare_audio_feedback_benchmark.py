"""Freeze thirty existing training cases and controlled offline meaning contrasts."""
import argparse
import hashlib
import json
import random
import re
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replace_attribute(text, original, replacement):
    pattern = r'\b' + re.escape(original) + r'\b'
    if len(re.findall(pattern, text)) != 1:
        raise ValueError('Minimal attribute replacement must target exactly one whole word')
    return re.sub(pattern, replacement, text)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--teachers', type=Path, required=True)
    p.add_argument('--labels', type=Path, required=True)
    p.add_argument('--contrast-spec', type=Path, required=True)
    p.add_argument('--design', type=Path, required=True)
    p.add_argument('--human-export', type=Path, required=True)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    if args.output_dir.exists():
        raise ValueError('Use fresh private benchmark directory')
    dataset = json.loads(args.dataset.read_text())
    teachers = json.loads(args.teachers.read_text())
    if dataset['labels_sha256'] != sha(args.labels) or teachers['dataset_sha256'] != sha(args.dataset):
        raise ValueError('Teacher/annotation provenance differs')
    labels = {r['id']: r for r in map(json.loads, args.labels.read_text().splitlines())}
    changes = json.loads(args.contrast_spec.read_text())['changes']
    design = {r['id']: r for r in json.loads(args.design.read_text())['rows']}
    targets = {str(Path(r['prefix_file']).parent):r for r in teachers['records']}
    training = [r for r in dataset['rows'] if r['split'] == 'train']
    if any(r['split'] not in {'train', 'heldout'} for r in dataset['rows']):
        raise ValueError('Reserved/fresh input forbidden')
    selected, global_groups = [], set()
    for category in sorted({design[r['id']]['scene'] for r in training}):
        pool = sorted([r for r in training if design[r['id']]['scene'] == category], key=lambda r:r['id'])
        random.Random(20261016).shuffle(pool)
        count = 0
        for row in pool:
            if row['scene_group'] in global_groups:
                continue
            selected.append(row);global_groups.add(row['scene_group']);count += 1
            if count == 3:
                break
        if count != 3:
            raise ValueError('Need three independent training capture groups per category')
    if len(selected) != 30 or {r['id'] for r in selected} != set(changes):
        raise ValueError('Frozen thirty-case identity selection changed')
    records = []
    for row in selected:
        key = row['id'];label = labels[key]
        if label.get('training_only') is not True or label['label'].get('keep') is not True:
            raise ValueError('Existing accepted training annotations only')
        text = label['label']['narration_it']
        original, replacement = changes[key]
        contrast = replace_attribute(text, original, replacement)
        record = targets[row['cache_dir']]
        for file, expected in [(record['prefix_file'],record['prefix_sha256']),
                               (record['audio'],record['audio_sha256']), (record['codes_file'],record['codes_sha256'])]:
            if sha(file) != expected:
                raise ValueError('Historical teacher artifact changed')
        records.append({'id':key, 'split':'train', 'scene':design[key]['scene'], 'scene_group':row['scene_group'],
            'image':design[key]['path'], 'image_sha256':sha(design[key]['path']),
            'source_text':text, 'attribute_text':contrast, 'negation_text':'È falso dire: '+text,
            'substitution':{'original':original,'replacement':replacement},
            'audio':record['audio'], 'audio_sha256':record['audio_sha256'],
            'prefix_file':record['prefix_file'], 'prefix_sha256':record['prefix_sha256'],
            'features':row['features'], 'offline_diagnostic_text_only':True})
    degraded = []
    for entry in json.loads(args.human_export.read_text()):
        row = entry['mapping']
        notes = [r['note'] for r in entry['ratings']]
        if any(any(word in note.casefold() for word in ['glibberish','gibberish']) for note in notes):
            if design[row['image_id']]['split'] != 'train':
                raise ValueError('Rated degraded controls must use existing training images')
            degraded.append({'id':row['clip_id'], 'source_image_id':row['image_id'], 'audio':row['audio'],
                'audio_sha256':sha(row['audio']), 'human_note':notes, 'human_quality_label':'explicit_partial_or_full_gibberish'})
    info = {'scope':'Offline training-only ASFL feasibility; texts are references, not runtime conditioning',
        'source_sha256':sha(__file__), 'dataset_sha256':sha(args.dataset), 'teacher_manifest_sha256':sha(args.teachers),
        'labels_sha256':sha(args.labels), 'contrast_spec_sha256':sha(args.contrast_spec), 'design_sha256':sha(args.design), 'protocol_sha256':sha(args.protocol),
        'human_export_sha256':sha(args.human_export), 'seed':20261016,
        'selection':'Three capture groups per each of ten scenes, shuffled with a fixed seed before scores',
        'primary_text_format':'Document', 'secondary_text_formats':['SentenceSimilarity','unprefixed'],
        'secondary_formats_do_not_change_preregistered_gate':True,
        'true_independent_speaker_validation':False,
        'records':records, 'human_rated_degraded':degraded}
    args.output_dir.mkdir(mode=0o700)
    file = args.output_dir/'benchmark.json';file.write_text(json.dumps(info,ensure_ascii=False,indent=2)+'\n');file.chmod(0o600)
    print(json.dumps({'training_cases':len(records),'scene_groups':len(global_groups),'attribute_pairs':len(records),
                      'human_rated_degraded':len(degraded),'new_neural_training':False}))


if __name__ == '__main__':
    main()
