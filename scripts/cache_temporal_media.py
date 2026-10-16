"""Cache frozen audio histories from a strict text-free manifest in the vision worker."""
import argparse
import inspect
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from latent_diaries.feature_provenance import checksum, verify_image_features
from latent_diaries.multimodal_feedback_encoder import FrozenFeedbackEncoder


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--image-cache', type=Path, action='append', required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    if not Path('/.dockerenv').exists() or args.output_dir.exists():
        raise ValueError('Managed vision worker and fresh output required')
    data = json.loads(args.inputs.read_text())
    if set(data) != {'scope','source_sha256','corpus_sha256','teacher_manifest_sha256','images','audio','pairs'}:
        raise ValueError('Strict media-only manifest required')
    if len(data['images']) != 12 or len(data['audio']) != 48 or len(data['pairs']) != 24:
        raise ValueError('Complete12-image/48-fragment/24-sequence cache only')
    for row in data['images']:
        if set(row) != {'id','split','scene_group','features'}:
            raise ValueError('Image features/provenance only; no linguistic fields')
    for row in data['audio']:
        if set(row) != {'id','audio','audio_sha256'} or checksum(row['audio']) != row['audio_sha256']:
            raise ValueError('Verified media-only audio records required')
    for row in data['pairs']:
        if set(row) != {'id','image_id','split','scene_group','history_audio_id','next_audio_id'}:
            raise ValueError('Pair identities/provenance only')
    evidence = verify_image_features(data['images'], args.image_cache)
    encoder = FrozenFeedbackEncoder('914f7f89142e33e77833254d9c9b90c3cef7303b', 'cuda', 'float32')
    vectors, traces = [], []
    for start in range(0,48,4):
        value, trace = encoder.encode_audio([r['audio'] for r in data['audio'][start:start+4]])
        vectors.append(value);traces.append(trace)
    audio = torch.cat(vectors)
    image = torch.tensor([r['features'] for r in data['images']], dtype=torch.float32)
    if audio.shape != (48,768) or not torch.allclose(audio.norm(dim=-1), torch.ones(48), atol=1e-5):
        raise ValueError('Normalized finite48x768 audio vectors required')
    args.output_dir.mkdir(mode=0o700)
    file = args.output_dir/'features.safetensors'
    save_file({'audio': audio, 'image': image}, str(file))
    result = {'scope': 'Frozen teacher-history cache; no text input, optimizer or temporal reader',
        'source_sha256': checksum(__file__), 'inputs_sha256': checksum(args.inputs),
        'encoder_source_sha256': checksum(inspect.getfile(FrozenFeedbackEncoder)),
        'model_revision': encoder.revision, 'vision_lock_sha256': checksum('/opt/vision/uv.lock'),
        'worker_lock_sha256': checksum('uv.lock'), 'dtype': 'float32',
        'image_feature_provenance': evidence, 'features_sha256': checksum(file),
        'image_ids': [r['id'] for r in data['images']], 'audio_ids': [r['id'] for r in data['audio']],
        'pairs': data['pairs'], 'audio_traces': traces, 'variable_text_used': False,
        'human_corpus_validation_pending': True, 'training_started': False}
    (args.output_dir/'manifest.json').write_text(json.dumps(result, indent=2)+'\n')
    print('Cached48 frozen audio vectors and verified12 historical image vectors; no training')


if __name__ == '__main__':
    main()
