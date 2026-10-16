"""Assemble only the prespecified coherent teacher sequences for blind corpus review."""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf

from latent_diaries.feature_provenance import checksum


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['corpus', 'teachers', 'output-dir']:
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--run-id', default='asfl-teacher-corpus-20261016')
    args = p.parse_args()
    if args.output_dir.exists() or not args.run_id or not all(c.isascii() and (c.isalnum() or c in '-_') for c in args.run_id):
        raise ValueError('Fresh output and safe review run ID required')
    corpus = json.loads(args.corpus.read_text());teachers = json.loads(args.teachers.read_text())
    if teachers['corpus_sha256'] != checksum(args.corpus) or len(teachers['records']) != 48:
        raise ValueError('All48 genuine teacher fragments must be complete before review')
    by_id = {r['id']: r for r in corpus['rows']}
    fragments = {r['id']: r for r in teachers['records']}
    ids = list(corpus['review_sequence_ids'])
    if len(ids) != len(set(ids)) or len(ids) != 8:
        raise ValueError('Exactly eight prespecified review sequences required')
    random.Random(2026101604).shuffle(ids)
    args.output_dir.mkdir(mode=0o700)
    manifest, mapping = [], []
    for i, key in enumerate(ids, 1):
        row = by_id[key]
        if checksum(row['image']) != row['image_sha256']:
            raise ValueError('Selected review pixels changed')
        waves = [];records = []
        for j in [1, 2]:
            target = fragments[key+f'-f{j}']
            if checksum(target['audio']) != target['audio_sha256'] or target['reached_cap']:
                raise ValueError('Changed/failed teacher fragment; do not replace the sequence')
            wave, rate = sf.read(target['audio'], dtype='float32')
            if rate != 24000 or wave.ndim != 1 or not np.isfinite(wave).all():
                raise ValueError('Finite mono24kHz genuine fragments required')
            waves.append(wave);records.append(target)
        pause = np.zeros(10800, dtype=np.float32)
        path = args.output_dir/f'TC-{i:02d}.wav'
        sf.write(path, np.concatenate([waves[0], pause, waves[1]]), 24000)
        clip_id = f'{args.run_id}-{i:02d}'
        manifest.append({'id': clip_id, 'run_id': args.run_id, 'blind_label': f'TC-{i:02d}',
                         'image': row['image'], 'audio': path.name})
        mapping.append({'clip_id': clip_id, 'sequence_id': key, 'split': row['split'],
                        'image_id': row['image_id'], 'audio_sha256': checksum(path),
                        'fragment_sources': records, 'pause_seconds': .45,
                        'second_fragment_starts_at': len(waves[0])/24000+.45})
    file = args.output_dir/'review.jsonl'
    file.write_text(''.join(json.dumps(r)+'\n' for r in manifest));file.chmod(0o600)
    private = args.output_dir/'private-mapping.json'
    private.write_text(json.dumps({'scope': 'Official TEXT-CONDITIONED teacher corpus validation, not textless student output',
        'source_sha256': checksum(__file__), 'corpus_sha256': checksum(args.corpus),
        'teacher_manifest_sha256': checksum(args.teachers), 'mapping': mapping,
        'human_ratings_created': 0, 'training_started': False}, indent=2)+'\n');private.chmod(0o600)
    print('Prepared8 blinded coherent teacher sequences; no human votes created')


if __name__ == '__main__':
    main()
