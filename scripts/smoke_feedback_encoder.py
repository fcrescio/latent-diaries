"""Exercise pinned full-media encoder and unchanged image-only parity; no training."""
import argparse
import inspect
import json
import os
import time
from pathlib import Path

import torch
import transformers
from PIL import Image, ImageOps
from safetensors.torch import load_file, save_file

from latent_diaries.multimodal_feedback_encoder import FrozenFeedbackEncoder
from latent_diaries.vision import FrozenImageEncoder, sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image', type=Path, required=True)
    p.add_argument('--audio', type=Path, required=True)
    p.add_argument('--image-id', required=True)
    p.add_argument('--image-cache', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--revision', default='914f7f89142e33e77833254d9c9b90c3cef7303b')
    p.add_argument('--dtype', choices=['float32', 'bfloat16'], default='float32')
    args = p.parse_args()
    if not Path('/.dockerenv').exists() or args.output_dir.exists():
        raise ValueError('Managed worker and fresh output required')
    manifest_file = args.image_cache / 'manifest.json'
    manifest = json.loads(manifest_file.read_text())
    weights = args.image_cache / 'features.safetensors'
    if sha256(weights) != manifest['features_sha256'] or manifest['variable_text_used'] is not False:
        raise ValueError('Original image cache changed')
    index = next(i for i, r in enumerate(manifest['records']) if r['id'] == args.image_id)
    if manifest['records'][index]['image_sha256'] != sha256(args.image):
        raise ValueError('Image differs from cached source')
    reference = load_file(str(weights))['embeddings'][index:index+1]
    with Image.open(args.image) as im:
        image = ImageOps.exif_transpose(im).convert('RGB')
    start = time.monotonic()
    encoder = FrozenFeedbackEncoder(args.revision, 'cuda', dtype=args.dtype)
    load_seconds = time.monotonic()-start
    torch.cuda.reset_peak_memory_stats()
    visual, visual_meta = encoder.encode_image([image])
    audio, audio_meta = encoder.encode_audio([args.audio])
    joint, joint_meta = encoder.encode_image_audio([image], [args.audio])
    audio_repeat, _ = encoder.encode_audio([args.audio])
    audio_batch, _ = encoder.encode_audio([args.audio, args.audio])
    joint_batch, _ = encoder.encode_image_audio([image, image], [args.audio, args.audio])
    metrics = {'image_cache_max_abs_error': float((visual-reference).abs().max()),
               'audio_repeat_max_abs_error': float((audio-audio_repeat).abs().max()),
               'audio_batch_max_abs_error': float((audio-audio_batch).abs().max()),
               'joint_batch_max_abs_error': float((joint-joint_batch).abs().max())}
    if args.dtype == 'float32':
        legacy = FrozenImageEncoder(args.revision, 'cuda')
        legacy_visual, _ = legacy.encode([image])
        metrics['live_legacy_max_abs_error'] = float((visual-legacy_visual).abs().max())
        if metrics['live_legacy_max_abs_error'] > 1e-6 or metrics['image_cache_max_abs_error'] > 1e-6:
            raise ValueError('Loading audio changed the legacy image path')
        del legacy
        if metrics['audio_repeat_max_abs_error'] > 1e-6 or max(metrics['audio_batch_max_abs_error'], metrics['joint_batch_max_abs_error']) > 1e-4:
            raise ValueError('Full-media repeat/batch parity failed')
    args.output_dir.mkdir(mode=0o700)
    feature_path = args.output_dir / 'features.safetensors'
    save_file({'image': visual, 'audio': audio, 'joint': joint}, str(feature_path))
    result = {'scope': 'Phase0 media encoder compatibility; no semantic quality or trained narrator claim',
        'model_id': 'google/embeddinggemma-2', 'model_revision': args.revision,
        'source_revision': os.environ.get('LD_CODE_REVISION'), 'source_sha256': sha256(Path(__file__)),
        'encoder_source_sha256': sha256(Path(inspect.getfile(FrozenFeedbackEncoder))),
        'upstream_model_source_sha256': sha256(Path(inspect.getfile(type(encoder.model)))),
        'upstream_processor_source_sha256': sha256(Path(inspect.getfile(type(encoder.processor)))),
        'audio_feature_extractor_source_sha256': sha256(Path(inspect.getfile(type(encoder.processor._audio_processor)))),
        'vision_lock_sha256': sha256(Path('/opt/vision/uv.lock')), 'worker_lock_sha256': sha256(Path('uv.lock')),
        'image_sha256': sha256(args.image), 'audio_sha256': sha256(args.audio),
        'image_cache_manifest_sha256': sha256(manifest_file), 'features_sha256': sha256(feature_path),
        'dtype': args.dtype, 'torch_version': torch.__version__, 'transformers_version': transformers.__version__,
        'parameter_count': sum(p.numel() for p in encoder.model.parameters()),
        'all_parameters_frozen': not any(p.requires_grad for p in encoder.model.parameters()),
        'audio_tower_loaded': encoder.model.audio_tower is not None,
        'vision_tower_loaded': encoder.model.vision_tower is not None,
        'load_seconds': load_seconds, 'elapsed_seconds': time.monotonic()-start,
        'peak_allocated_vram_bytes': torch.cuda.max_memory_allocated(), 'metrics': metrics,
        'modalities': {'image': visual_meta, 'audio': audio_meta, 'joint': joint_meta},
        'norms': [float(v.norm()) for v in [visual, audio, joint]], 'variable_text_used': False}
    (args.output_dir / 'manifest.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
