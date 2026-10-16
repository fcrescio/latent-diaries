"""Confirm frozen semantic retrieval on existing teachers, without fitting or fresh data."""
import argparse
import json
from pathlib import Path

import numpy as np
from cache_audio_feedback_features import offline_text_reference
from safetensors.numpy import save_file

from latent_diaries.audio_feedback_metrics import cluster_mean_interval, duplicate_aware_hits
from latent_diaries.feature_provenance import checksum
from latent_diaries.multimodal_feedback_encoder import FrozenFeedbackEncoder


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['dataset', 'teachers', 'labels', 'pilot', 'output-dir']:
        p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args()
    if not Path('/.dockerenv').exists() or args.output_dir.exists():
        raise ValueError('Managed worker and fresh private output required')
    dataset = json.loads(args.dataset.read_text())
    teachers = json.loads(args.teachers.read_text())
    pilot = json.loads(args.pilot.read_text())
    if (teachers['dataset_sha256'] != checksum(args.dataset)
            or dataset['labels_sha256'] != checksum(args.labels)
            or pilot['dataset_sha256'] != checksum(args.dataset)):
        raise ValueError('Historical input provenance differs')
    labels = {r['id']: r for r in map(json.loads, args.labels.read_text().splitlines())}
    targets = {str(Path(r['prefix_file']).parent): r for r in teachers['records']}
    rows = [r for r in dataset['rows'] if r['split'] == 'train']
    if len(rows) != len(teachers['records']) or any(r['split'] not in {'train', 'heldout'} for r in dataset['rows']):
        raise ValueError('Only complete existing training teachers are permitted')
    texts, paths = [], []
    for row in rows:
        label = labels[row['id']]
        if label.get('training_only') is not True or label['label'].get('keep') is not True:
            raise ValueError('Existing approved training annotations required')
        target = targets[row['cache_dir']]
        if checksum(target['audio']) != target['audio_sha256']:
            raise ValueError('Historical teacher audio changed')
        texts.append(label['label']['narration_it'])
        paths.append(target['audio'])
    args.output_dir.mkdir(mode=0o700)
    encoder = FrozenFeedbackEncoder('914f7f89142e33e77833254d9c9b90c3cef7303b', 'cuda', 'float32')
    audio, text, traces = [], [], []
    for start in range(0, len(rows), 4):
        vectors, trace = encoder.encode_audio(paths[start:start+4])
        audio.extend(vectors.numpy());traces.append(trace)
        text.extend(offline_text_reference(encoder, texts[start:start+4], 'Document').numpy())
        print(json.dumps({'completed': min(start+4, len(rows)), 'total': len(rows)}), flush=True)
    audio, text = np.asarray(audio), np.asarray(text)
    image = np.asarray([r['features'] for r in rows], dtype=np.float32)
    if image.shape != audio.shape or not np.allclose(np.linalg.norm(image, axis=-1), 1, atol=1e-5):
        raise ValueError('Historical normalized image vectors differ')
    features = args.output_dir/'features.safetensors'
    save_file({'audio': audio, 'text': text, 'image': image}, str(features))
    scores, visual_scores = audio@text.T, image@audio.T
    pilot_ids = {r['id'] for r in pilot['records']}
    results = {}
    for name, indices in [('all', list(range(len(rows)))),
                          ('pilot', [i for i, r in enumerate(rows) if r['id'] in pilot_ids]),
                          ('additional', [i for i, r in enumerate(rows) if r['id'] not in pilot_ids])]:
        groups = [rows[i]['scene_group'] for i in indices]
        strict, visual = [], []
        for i in indices:
            others = np.arange(len(rows)) != i
            strict.append(bool(scores[i, i] > scores[i, others].max()))
            visual.append(bool(visual_scores[i, i] > visual_scores[i, others].max()))
        results[name] = {
            'strict_audio_text_top1': cluster_mean_interval(strict, groups),
            'duplicate_reference_aware_audio_text_top1': cluster_mean_interval(
                duplicate_aware_hits(scores[indices], texts, indices), groups),
            'strict_image_audio_top1': cluster_mean_interval(visual, groups),
            'fixed_catalog_size': len(rows)}
    report = {'scope': 'Offline confirmation on existing training teachers; no training, threshold tuning or fresh-image access',
              'source_sha256': checksum(__file__),
              'inputs': {name: checksum(getattr(args, name)) for name in ['dataset', 'teachers', 'labels', 'pilot']},
              'model_revision': encoder.revision, 'dtype': 'float32',
              'cases': len(rows), 'unique_reference_strings': len(set(texts)),
              'features_sha256': checksum(features), 'row_ids': [r['id'] for r in rows],
              'results': results, 'audio_traces': traces,
              'human_quality_verified': False, 'independent_image_generalization_verified': False}
    (args.output_dir/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    main()
