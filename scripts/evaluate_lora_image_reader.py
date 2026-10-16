"""Strict image-only inference for a reader and matching final-attention LoRA."""
import argparse
import json
from pathlib import Path

import torch
from image_probe import make_reader
from safetensors.torch import load_file

from latent_diaries.image_reader import validate_image_inputs
from latent_diaries.inversion import digest, load_backbone, render
from latent_diaries.talker_lora import inject_talker_lora, load_lora_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--ids', nargs='+', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--codec0-greedy', action='store_true')
    args = parser.parse_args()
    if not Path('/.dockerenv').exists() or args.output_dir.exists():
        raise ValueError('Managed worker and fresh output required')
    info = json.loads((args.run_dir / 'config.json').read_text())
    for name in ['reader', 'lora']:
        if digest(args.run_dir / (name+'.safetensors')) != info[name+'_sha256']:
            raise ValueError('Reader/LoRA checksum differs')
    if (info['lock_sha256'] != digest(Path('uv.lock')) or info['talker_lora_rank'] != 8
            or info['lora_source_sha256'] != digest(Path('src/latent_diaries/talker_lora.py'))):
        raise ValueError('LoRA protocol/dependency/source mismatch')
    rows = json.loads(args.inputs.read_text())['rows']
    validate_image_inputs(rows, info['train_ids']+info['heldout_ids'])
    selected = [r for r in rows if r['id'] in args.ids]
    if {r['id'] for r in selected} != set(args.ids):
        raise ValueError('Requested IDs absent')
    wrapper, builder, speaker = load_backbone({**info, 'talker_lora_rank': 0})
    modules = inject_talker_lora(wrapper.model.talker, rank=info['talker_lora_rank'], layers=info['talker_lora_layers'])
    if modules != info['lora_modules']:
        raise ValueError('LoRA projection layout mismatch')
    load_lora_state(wrapper.model.talker, load_file(str(args.run_dir/'lora.safetensors')))
    wrapper.model.requires_grad_(False).eval()
    reader = make_reader(info, load_file(str(args.run_dir/'reader.safetensors'))).to(speaker.device).eval()
    args.output_dir.mkdir(parents=True)
    selected.append({'id':'null', 'split':'control', 'features':[0.0]*768})
    results = []
    for row in selected:
        with torch.no_grad():
            prefix, _, length = reader(torch.tensor([row['features']], device=speaker.device))
        output = render(builder, wrapper, prefix[:, :int(length[0])], speaker,
                        args.output_dir/f"{row['id']}.wav", 160, 42, codec0_greedy=args.codec0_greedy)
        results.append({'id':row['id'], 'split':row['split'], **output})
    (args.output_dir/'results.json').write_text(json.dumps({
        'results':results, 'reader_sha256':info['reader_sha256'], 'lora_sha256':info['lora_sha256'],
        'source_sha256':digest(Path(__file__)), 'inputs_sha256':digest(args.inputs),
        'text_used':False, 'teacher_files_read':False, 'linguistic_labels_read':False,
        'scope':'Exposed development generation; human quality/grounding not inferred'}, indent=2)+'\n')


if __name__ == '__main__':
    main()
