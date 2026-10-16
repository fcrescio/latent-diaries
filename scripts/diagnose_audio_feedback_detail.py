"""Post-pilot visual detail and isolated audio parity checks on sealed caches."""
import argparse
import json
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

from latent_diaries.audio_feedback_metrics import cluster_mean_interval
from latent_diaries.feature_provenance import checksum


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('Fresh diagnostic output required')
    manifest = json.loads((args.root/'features/manifest.json').read_text())
    def load(name):
        record = manifest['groups'][name]
        if checksum(record['file']) != record['sha256']:
            raise ValueError('Sealed feature tensor changed')
        return load_file(record['file'])['embeddings'].astype(np.float64)
    image, original, changed = [load(k) for k in ['image', 'audio_clean', 'audio_attribute']]
    benchmark = json.loads((args.root/'benchmark/benchmark.json').read_text())
    groups = [r['scene_group'] for r in benchmark['records']]
    margin = (image*(original-changed)).sum(-1)
    isolation_path = args.root/'audio-only-isolation/manifest.json'
    isolation = json.loads(isolation_path.read_text())
    # The independent process has only media paths/checksums, not benchmark labels.
    tensors = load_file(str(args.root/'audio-only-isolation/features.safetensors'))
    results = {
        'scope': 'Post-pilot diagnostics only; no changed thresholds, fitting or human quality claims',
        'source_sha256': checksum(__file__),
        'feature_manifest_sha256': checksum(args.root/'features/manifest.json'),
        'isolation_manifest_sha256': checksum(isolation_path),
        'visual_preference_original_over_attribute_change': cluster_mean_interval(margin > 0, groups),
        'visual_original_minus_attribute_cosine': cluster_mean_interval(margin, groups),
        'isolated_tensor_keys': list(tensors),
        'isolation_metadata': {k: v for k, v in isolation.items() if isinstance(v, (bool, int, float))}}
    for key, values in tensors.items():
        if values.shape == original.shape:
            results['isolated_'+key+'_max_abs_error'] = float(np.abs(values-original).max())
    args.output.write_text(json.dumps(results, indent=2)+'\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
