"""Export numeric/media-only teacher-history inputs; strip all linguistic supervision."""
import argparse
import json
from pathlib import Path

from latent_diaries.feature_provenance import checksum


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['corpus', 'teachers', 'output']:
        p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('Fresh media-only export required')
    corpus = json.loads(args.corpus.read_text());teachers = json.loads(args.teachers.read_text())
    if teachers['corpus_sha256'] != checksum(args.corpus) or len(teachers['records']) != 48:
        raise ValueError('Frozen complete teachers required')
    images, pairs = {}, []
    for row in corpus['rows']:
        images[row['image_id']] = {k: row[k] for k in ['split','scene_group','features']}
        images[row['image_id']]['id'] = row['image_id']
        pairs.append({'id': row['id'], 'image_id': row['image_id'], 'split': row['split'],
                      'scene_group': row['scene_group'], 'history_audio_id': row['id']+'-f1',
                      'next_audio_id': row['id']+'-f2'})
    audio = [{k: row[k] for k in ['id','audio','audio_sha256']} for row in teachers['records']]
    for row in audio:
        if checksum(row['audio']) != row['audio_sha256']:
            raise ValueError('Teacher waveform changed')
    result = {'scope': 'Teacher-history media cache only; no learned reader or text inference',
        'source_sha256': checksum(__file__), 'corpus_sha256': checksum(args.corpus),
        'teacher_manifest_sha256': checksum(args.teachers), 'images': list(images.values()),
        'audio': audio, 'pairs': pairs}
    args.output.write_text(json.dumps(result, indent=2)+'\n');args.output.chmod(0o600)
    print('Exported12 numeric image vectors and48 media-only audio records; no captions or transcripts')


if __name__ == '__main__':
    main()
