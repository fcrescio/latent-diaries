"""Standalone numeric-only prefix-flow inference; mark text-feature controls honestly."""
import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from latent_diaries.image_reader import validate_image_inputs
from latent_diaries.inversion import digest, load_backbone, render
from latent_diaries.prefix_flow import PrefixFlow


def load_flow(weights, width):
    model = PrefixFlow(*(weights[k] for k in ['mean', 'scale', 'center', 'basis', 'latent_scale']), width=width)
    model.load_state_dict(weights)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--ids', nargs='+', required=True)
    parser.add_argument('--mode', choices=['image', 'training-text-control'], required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if not Path('/.dockerenv').exists() or args.output_dir.exists():
        raise ValueError('Use managed worker and a fresh output directory')
    info = json.loads((args.run_dir / 'config.json').read_text())
    file = args.run_dir / 'reader.safetensors'
    if digest(file) != info['reader_sha256']:
        raise ValueError('Frozen reader checksum differs')
    inputs = json.loads(args.inputs.read_text())['rows']
    validate_image_inputs(inputs, [r['id'] for r in inputs], allow_new=True)
    if len(set(args.ids)) != len(args.ids) or not set(args.ids) <= {r['id'] for r in inputs}:
        raise ValueError('Requested IDs missing or duplicated')
    model = load_flow(load_file(str(file)), info['width'])
    wrapper, builder, speaker = load_backbone(info)
    model.to(speaker.device).eval().requires_grad_(False)
    chosen = [r for r in inputs if r['id'] in args.ids]
    chosen.append({'id': 'null', 'split': 'control', 'features': [0.0] * 768})
    args.output_dir.mkdir(parents=True)
    records = []
    for row in chosen:
        features = torch.tensor([row['features']], device=speaker.device, dtype=torch.float32)
        prefix, lengths = model.sample(features, seed=42, steps=64)
        length = int(lengths[0])
        generated = render(builder, wrapper, prefix[:, :length], speaker,
                           args.output_dir / f"{row['id']}.wav", max_frames=105, seed=42)
        record = {'id': row['id'], 'split': row['split'], 'predicted_prefix_length': length,
                  'human_listening_status': 'pending', **generated}
        records.append(record);print(json.dumps(record), flush=True)
    if digest(file) != info['reader_sha256']:
        raise ValueError('Inference changed weights')
    text_control = args.mode != 'image'
    result = {'scope': ('Text-derived embedding diagnostic, not textless image inference' if text_control
                        else 'Existing exposed-development images; no fresh-image acceptance'),
              'reader_sha256': info['reader_sha256'], 'inputs_sha256': digest(args.inputs),
              'source_sha256': digest(Path(__file__)), 'mode': args.mode,
              'image_conditioning': not text_control, 'text_used': text_control,
              'linguistic_labels_read': False, 'teacher_files_read': False, 'results': records}
    (args.output_dir / 'results.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
