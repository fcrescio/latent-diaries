"""Import the prespecified blinded no-training selection comparison; no votes."""
import argparse
import json
import random
from pathlib import Path

from latent_diaries.feature_provenance import checksum
from latent_diaries.review import ingest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--design',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--run-id',required=True)
    args = p.parse_args()
    if args.output_dir.exists() or not all(c.isascii() and (c.isalnum() or c in '-_') for c in args.run_id):
        raise ValueError('Fresh private output and safe run ID required')
    result = json.loads(args.selection.read_text());plan = json.loads(args.protocol.read_text())
    if result['protocol_sha256'] != checksum(args.protocol) or any(result[k] for k in ['text_used','teacher_files_read','linguistic_labels_read','asr_used']):
        raise ValueError('Verified textless media selection required')
    design = {r['id']:r for r in json.loads(args.design.read_text())['rows']}
    by_id = {r['id']:r for r in result['results']}
    pairs = []
    for key in plan['development_ids']:
        if design[key]['split']!='heldout':
            raise ValueError('Existing development images only')
        for method in ['first_candidate','fixed_random','image_audio_max']:
            pairs.append({'image_id':key,'method':method,**by_id[key]['methods'][method]})
    for key in plan['development_ids'][:4]:
        pairs.append({'image_id':key,'method':'shuffled_image_scorer',**by_id[key]['methods']['shuffled_image_scorer']})
    if len(pairs)!=28 or any(p['audio'] is None for p in pairs):
        raise ValueError('Prespecified28 comparison pairs incomplete; preserve/report failures, never fill them')
    for pair in pairs:
        if checksum(pair['audio'])!=pair['audio_sha256']:
            raise ValueError('Selected waveform changed')
    random.Random(2026101603).shuffle(pairs)
    args.output_dir.mkdir(mode=0o700)
    manifest,private = args.output_dir/'review.jsonl',args.output_dir/'private-mapping.json'
    rows,mapping=[],[]
    for i,pair in enumerate(pairs,1):
        row={'id':f'{args.run_id}-{i:02d}','run_id':args.run_id,'blind_label':f'AF-{i:02d}',
             'image':str(Path(design[pair['image_id']]['path']).resolve()),'audio':str(Path(pair['audio']).resolve())}
        rows.append(row);mapping.append({'clip_id':row['id'],'blind_label':row['blind_label'],**pair})
    manifest.write_text(''.join(json.dumps(r)+'\n'for r in rows));manifest.chmod(0o600)
    private.write_text(json.dumps({'scope':'No-training candidate selection comparison; no assumed quality improvement',
        'selection_sha256':checksum(args.selection),'protocol_sha256':checksum(args.protocol),'mapping':mapping},indent=2)+'\n');private.chmod(0o600)
    count=ingest(manifest,Path('run-data/reviews.sqlite3'),Path('run-data/review-assets'))
    print(f'Imported {count} genuine pairs including all duplicate selections; zero ratings created')


if __name__=='__main__':
    main()
