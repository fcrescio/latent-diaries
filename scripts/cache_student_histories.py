"""Local training rollouts and frozen-teacher targets; never publish as speech."""
import argparse
import json
import random
from pathlib import Path

import torch
from image_probe import make_reader
from safetensors.torch import load_file, save_file

from latent_diaries.inversion import condition, digest, load_backbone, trajectory_logits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--design', type=Path, required=True)
    parser.add_argument('--teachers', type=Path, required=True)
    parser.add_argument('--reader-run', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if not Path('/.dockerenv').exists() or args.output_dir.exists():
        raise ValueError('Managed worker and fresh rollout cache required')
    info = json.loads((args.reader_run/'config.json').read_text())
    if info.get('talker_lora_rank', 0) or info['dataset_sha256'] != digest(args.dataset):
        raise ValueError('Frozen matched reader required')
    weights = args.reader_run/'reader.safetensors'
    if digest(weights) != info['reader_sha256']:
        raise ValueError('Reader changed')
    rows = json.loads(args.dataset.read_text())['rows']
    if any(r['split'] not in {'train', 'heldout'} for r in rows):
        raise ValueError('No reserved/fresh access')
    design = {r['id']:r for r in json.loads(args.design.read_text())['rows']}
    grouped = {}
    for row in rows:
        if row['split'] == 'train':
            grouped.setdefault(design[row['id']]['scene'], []).append(row)
    selected = []
    for scene, values in sorted(grouped.items()):
        values = sorted(values, key=lambda r:r['id'])
        random.Random(42).shuffle(values)
        seen = set()
        for row in values:
            if row['scene_group'] not in seen:
                selected.append(row);seen.add(row['scene_group'])
            if len(seen) == 6:
                break
    selected = sorted(selected, key=lambda r:r['id'])
    manifest = json.loads((args.teachers/'manifest.json').read_text())
    if manifest['dataset_sha256'] != digest(args.dataset) or manifest['lock_sha256'] != digest(Path('uv.lock')):
        raise ValueError('Teacher provenance differs')
    by_key = {str(Path(r['prefix_file']).parent):r for r in manifest['records']}
    wrapper, builder, speaker = load_backbone(info)
    reader = make_reader(info, load_file(str(weights))).to(speaker.device).eval()
    args.output_dir.mkdir(parents=True)
    records = []
    with torch.inference_mode():
        for start in range(0, len(selected), 8):
            batch = selected[start:start+8]
            prefix = reader(torch.tensor([r['features'] for r in batch], device=speaker.device))[0]
            # Only image-derived vectors and fixed voice/protocol condition this generation.
            histories = builder.generate_codes(condition(prefix, speaker.expand(len(batch), -1)),
                max_frames=160, seed=args.seed+start, codec0_greedy=True)
            for row, codes in zip(batch, histories, strict=True):
                if not len(codes):
                    raise ValueError('Empty student history; inspect before fitting')
                teacher = by_key[row['cache_dir']]
                if digest(Path(teacher['prefix_file'])) != teacher['prefix_sha256']:
                    raise ValueError('Teacher prefix changed')
                # Teacher supervision is consulted only after standalone student generation.
                teacher_prefix = load_file(teacher['prefix_file'])['soft_tokens'].to(speaker.device)
                first, rest = trajectory_logits(wrapper, builder, teacher_prefix, speaker, codes[None])
                if not torch.isfinite(first).all() or not torch.isfinite(rest).all():
                    raise ValueError('Nonfinite teacher rollout targets')
                path = args.output_dir/f"history-{row['id']}.safetensors"
                save_file({'audio_codes':codes[None].cpu().contiguous(),
                           'codec0_logits':first.cpu().contiguous(),
                           'residual_logits':rest.cpu().contiguous()},str(path))
                records.append({'id':row['id'], 'prefix_key':row['cache_dir'], 'file':str(path),
                                'sha256':digest(path), 'frames':len(codes), 'reached_cap':len(codes)==160})
            print(json.dumps({'histories':len(records),'total':len(selected)}),flush=True)
    output = {'records':records, 'dataset_sha256':digest(args.dataset),
        'teacher_manifest_sha256':digest(args.teachers/'manifest.json'),
        'source_reader_sha256':digest(weights), 'source_reader_config_sha256':digest(args.reader_run/'config.json'),
        'lock_sha256':digest(Path('uv.lock')), 'source_sha256':digest(Path(__file__)),
        'seed':args.seed,'batch_size':8,'maximum_frames':160,'codec0_greedy':True,
        'scope':'Training-only student-generated histories; no waveform/quality claim, no heldout labels'}
    (args.output_dir/'manifest.json').write_text(json.dumps(output,indent=2)+'\n')


if __name__ == '__main__':
    main()
