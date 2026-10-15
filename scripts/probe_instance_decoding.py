"""Textless duration/sampling controls for a fitted image reader."""
import argparse
import json
from pathlib import Path

import torch
from image_probe import make_reader
from safetensors.torch import load_file

from latent_diaries.image_reader import validate_image_inputs
from latent_diaries.inversion import digest, load_backbone, render


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--ids', nargs='+', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--max-frames', type=int, default=160)
    parser.add_argument('--codec0-greedy', action='store_true')
    args = parser.parse_args()
    if not Path('/.dockerenv').exists() or args.output_dir.exists():
        raise ValueError('Managed worker and fresh output required')
    info = json.loads((args.run_dir / 'config.json').read_text())
    weights = args.run_dir / 'reader.safetensors'
    if digest(weights) != info['reader_sha256'] or not 1 <= args.max_frames <= 160:
        raise ValueError('Weights/cap differ from supported control')
    rows = json.loads(args.inputs.read_text())['rows']
    validate_image_inputs(rows, info['train_ids']+info['heldout_ids'])
    selected = [r for r in rows if r['id'] in args.ids]
    if {r['id'] for r in selected} != set(args.ids):
        raise ValueError('Requested IDs absent')
    wrapper, builder, speaker = load_backbone(info)
    reader = make_reader(info, load_file(str(weights))).to(speaker.device).eval()
    args.output_dir.mkdir(parents=True)
    results = []
    for row in selected:
        with torch.no_grad():
            prefix, _, length = reader(torch.tensor([row['features']], device=speaker.device))
        result = render(builder, wrapper, prefix[:, :int(length[0])], speaker,
                        args.output_dir / f"{row['id']}.wav", args.max_frames, 42,
                        codec0_greedy=args.codec0_greedy)
        results.append({'id': row['id'], 'split': row['split'], **result})
    (args.output_dir / 'results.json').write_text(json.dumps({
        'results': results, 'reader_sha256': digest(weights), 'inputs_sha256': digest(args.inputs),
        'source_sha256': digest(Path(__file__)), 'maximum_frames': args.max_frames,
        'text_used': False, 'teacher_files_read': False, 'linguistic_labels_read': False,
        'scope': 'Development duration/sampling diagnostic, not quality acceptance'}, indent=2)+'\n')


if __name__ == '__main__':
    main()
