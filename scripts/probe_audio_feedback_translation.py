"""Offline cross-language semantic probe using a private, separately supplied spec."""
import argparse
import json
from pathlib import Path

import numpy as np
from cache_audio_feedback_features import offline_text_reference
from safetensors.numpy import load_file, save_file

from latent_diaries.audio_feedback_metrics import minimal_pair, retrieval
from latent_diaries.feature_provenance import checksum
from latent_diaries.multimodal_feedback_encoder import FrozenFeedbackEncoder


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['benchmark', 'features-dir', 'reference-spec', 'output-dir']:
        p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args()
    if not Path('/.dockerenv').exists() or args.output_dir.exists():
        raise ValueError('Managed worker and fresh private output required')
    benchmark = json.loads(args.benchmark.read_text())
    references = json.loads(args.reference_spec.read_text())
    manifest = json.loads((args.features_dir/'manifest.json').read_text())
    rows = benchmark['records'];pairs = references['records']
    if (manifest['benchmark_sha256'] != checksum(args.benchmark)
            or [r['id'] for r in rows] != [r['id'] for r in pairs]):
        raise ValueError('Sealed benchmark identities/provenance differ')
    encoder = FrozenFeedbackEncoder('914f7f89142e33e77833254d9c9b90c3cef7303b', 'cuda', 'float32')
    texts = [r['translation'] for r in pairs]+[r['attribute_translation'] for r in pairs]
    features = np.concatenate([offline_text_reference(encoder, texts[i:i+4], 'Document').numpy()
                               for i in range(0, len(texts), 4)])
    def load(name):
        record = manifest['groups'][name]
        if checksum(record['file']) != record['sha256']:
            raise ValueError('Sealed media features changed')
        return load_file(record['file'])['embeddings']
    original, changed = load('audio_clean'), load('audio_attribute')
    groups = [r['scene_group'] for r in rows];n = len(rows)
    args.output_dir.mkdir(mode=0o700)
    file = args.output_dir/'features.safetensors';save_file({'offline_text': features}, str(file))
    report = {'scope': 'Offline cross-language probe, not inference, training or threshold tuning',
              'source_sha256': checksum(__file__), 'benchmark_sha256': checksum(args.benchmark),
              'reference_spec_sha256': checksum(args.reference_spec),
              'media_manifest_sha256': checksum(args.features_dir/'manifest.json'),
              'features_sha256': checksum(file), 'model_revision': encoder.revision,
              'translation_human_validated': references['human_translation_validation'],
              'audio_to_translated_text': retrieval(original@features[:n].T, groups),
              'translated_attribute_pairs': minimal_pair(original, changed, features[:n], features[n:], groups),
              'limitation': 'Assistant-authored translations may alter nuances; multilingual retrieval alone does not establish narrative memory or acceptable speech.'}
    (args.output_dir/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k.endswith(('text', 'pairs'))}, indent=2))


if __name__ == '__main__':
    main()
