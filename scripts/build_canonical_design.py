"""Freeze canonical training/development rows; fresh reserved groups stay unopened."""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

from latent_diaries.canonical_supervision import validate_scene_label


def select_training(rows):
    # Preserve the original cohort first, then add deterministic broader coverage.
    old = [r for r in rows if r['cohort']=='old']
    new = [r for r in rows if r['cohort']!='old']
    random.Random(42).shuffle(new)
    groups, classes, selected = Counter(), Counter(), []
    for row in old + new:
        if groups[row['scene_group']] >= 2 or classes[row['scene']] >= 48:
            continue
        selected.append(row)
        groups[row['scene_group']] += 1
        classes[row['scene']] += 1
    active = {scene for scene,n in classes.items() if n >= 8}
    return [r for r in selected if r['scene'] in active], active


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--canonical-texts', type=Path, required=True)
    parser.add_argument('--archive-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError('Use a fresh frozen design directory')
    canonical = json.loads(args.canonical_texts.read_text())
    labels = {r['id']: validate_scene_label(r['label']) for r in
              [json.loads(line) for line in args.labels.read_text().splitlines() if line]}
    rows = []
    for row in json.loads(args.inventory.read_text()):
        if row['split']=='fresh_test':
            raise ValueError('Reserved fresh rows must not enter development preparation')
        label = labels.get(row['id'])
        if not label or not label['keep']:
            continue
        scene = label['scene']
        rows.append(row | {'scene':scene,'teacher_text':canonical['texts'][scene]})
    train, active = select_training([r for r in rows if r['split']=='train'])
    dev = [r | {'split':'heldout'} for r in rows
           if r['split']=='development' and r['scene'] in active]
    if len(train) < 80 or len(dev) < 12:
        raise ValueError('Insufficient supported training/development cases')
    assert not {r['scene_group'] for r in train} & {r['scene_group'] for r in dev}
    chosen = sorted(train+dev,key=lambda r:r['id'])
    args.output_dir.mkdir(parents=True,mode=0o700)
    (args.output_dir/'design.json').write_text(json.dumps({'rows':chosen,'active_classes':sorted(active),
        'prefix_length':canonical['prefix_length'],'scope':'development; fresh data not accessed'},indent=2)+'\n')
    # Old vectors are already cached; extract only genuinely new selected images.
    manifest = [{'id':r['id'],'image':str(Path('/private/photos')/Path(r['path']).relative_to(args.archive_root)),
                 'scene_group':r['scene_group'],'source':'user-authorized local archive',
                 'license':'Local tests/training only; no publication/upload'}
                for r in chosen if r['cohort']!='old']
    (args.output_dir/'new-images.json').write_text(json.dumps(manifest,indent=2)+'\n')
    summary = {'training':len(train),'old_training':sum(r['cohort']=='old' for r in train),
               'development':len(dev),'active_class_count':len(active),
               'training_class_counts':dict(Counter(r['scene'] for r in train)),
               'development_class_counts':dict(Counter(r['scene'] for r in dev))}
    (args.output_dir/'private-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    for p in args.output_dir.iterdir():p.chmod(0o600)
    print(json.dumps({k:v for k,v in summary.items() if not isinstance(v,dict)}),flush=True)


if __name__ == '__main__':
    main()
