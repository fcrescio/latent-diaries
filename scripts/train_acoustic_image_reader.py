"""Bounded warm-start image reader training; Qwen and vision stay frozen."""
import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
from image_probe import make_reader, save
from safetensors.torch import load_file

from latent_diaries.acoustic_reader import acoustic_objective
from latent_diaries.inversion import digest, load_backbone


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--teachers', type=Path, required=True)
    parser.add_argument('--initial-run', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--steps', type=int, required=True)
    parser.add_argument('--prefix-weight', type=float, default=0.1)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if not Path('/.dockerenv').exists() or not 1 <= args.steps <= 320:
        raise ValueError('Use managed worker, within fixed 320-step schedule')
    if not math.isfinite(args.prefix_weight) or args.prefix_weight < 0:
        raise ValueError('Finite nonnegative prefix weight required')
    if args.run_dir.exists() and not args.resume:
        raise ValueError('Use a fresh run directory or explicit resume')
    info = json.loads((args.initial_run / 'config.json').read_text())
    initial_path = args.initial_run / 'reader.safetensors'
    if digest(initial_path) != info['reader_sha256'] or digest(args.dataset) != info['dataset_sha256']:
        raise ValueError('Initial model/dataset fingerprint differs')
    if info['evaluation_scope'] != 'development':
        raise ValueError('Expected constant-length development initialization')
    rows = json.loads(args.dataset.read_text())['rows']
    if any(r['split'] not in {'train', 'heldout'} for r in rows):
        raise ValueError('Fresh-test access forbidden')
    training = [r for r in rows if r['split'] == 'train']
    if {r['scene_group'] for r in training} & {r['scene_group'] for r in rows if r['split'] == 'heldout'}:
        raise ValueError('Scene-group leakage')
    teacher_manifest = args.teachers / 'manifest.json'
    teachers = json.loads(teacher_manifest.read_text())
    if teachers['dataset_sha256'] != digest(args.dataset) or teachers['lock_sha256'] != digest(Path('uv.lock')):
        raise ValueError('Teacher dataset/lock differs')
    info.update(initial_reader_sha256=digest(initial_path), completed_steps=0,
                teacher_manifest_sha256=digest(teacher_manifest),
                initial_config_sha256=digest(args.initial_run / 'config.json'),
                source_sha256=digest(Path(__file__)),
                acoustic_source_sha256=digest(Path('src/latent_diaries/acoustic_reader.py')),
                conditioning_source_sha256=digest(Path('src/latent_diaries/conditioning.py')),
                inversion_source_sha256=digest(Path('src/latent_diaries/inversion.py')),
                source_revision=os.environ.get('LD_CODE_REVISION'), learning_rate=2e-5,
                schedule_steps=320, warmup_steps=20,
                prefix_weight=args.prefix_weight,
                objective=f'codec0 CE + 0.1 residual CE + {args.prefix_weight} normalized prefix MSE',
                scope='Warm-start acoustic image-reader development control; backbones frozen')
    if args.resume:
        saved = json.loads((args.run_dir / 'config.json').read_text())
        for key, value in info.items():
            if key not in {'completed_steps', 'reader_sha256', 'source_revision',
                           'peak_allocated_vram_bytes'} and saved[key] != value:
                raise ValueError(f'Resume configuration differs: {key}')
        info = saved
    wrapper, builder, speaker = load_backbone(info)
    for key in ['model_id', 'model_revision']:
        if teachers[key] != info[key]:
            raise ValueError('Teacher backbone provenance differs')
    if list(builder.role_ids) != teachers['role_ids']:
        raise ValueError('Teacher/student role protocol differs')
    reader = make_reader(info, load_file(str(initial_path))).to(speaker.device).train()
    optimizer = torch.optim.AdamW(reader.parameters(), lr=2e-5, weight_decay=0)
    targets = {}
    for record in teachers['records']:
        for name in ['prefix', 'codes']:
            if digest(Path(record[f'{name}_file'])) != record[f'{name}_sha256']:
                raise ValueError('Teacher tensor changed')
        gold = load_file(record['codes_file'])['audio_codes'].to(speaker.device)
        if gold.ndim != 3 or gold.shape[0] != 1 or gold.shape[-1] != 16 or gold.dtype != torch.long:
            raise ValueError('Expected gold [1,F,16] integer codes')
        targets[str(Path(record['prefix_file']).parent)] = (
            load_file(record['prefix_file'])['soft_tokens'].to(speaker.device), gold)
    groups = {key: [r for r in training if r['cache_dir'] == key] for key in sorted(targets)}
    if any(not group for group in groups.values()) or sum(map(len, groups.values())) != len(training):
        raise ValueError('Teacher/class training coverage differs')
    features = {r['id']: torch.tensor([r['features']], dtype=torch.float32, device=speaker.device)
                for r in training}
    keys = sorted(groups)
    start = 0
    torch.manual_seed(42)
    if args.resume:
        if digest(args.run_dir / 'reader.safetensors') != info['reader_sha256']:
            raise ValueError('Reader checksum differs')
        reader.load_state_dict(load_file(str(args.run_dir / 'reader.safetensors')))
        state = torch.load(args.run_dir / 'state.pt', weights_only=True, map_location=speaker.device)
        if state['reader_sha256'] != info['reader_sha256'] or state['step'] != info['completed_steps']:
            raise ValueError('Checkpoint pair inconsistent')
        optimizer.load_state_dict(state['optimizer'])
        start = state['step']
        torch.set_rng_state(state['cpu_rng'].cpu())
        if state['cuda_rng']:
            torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda_rng']])
    args.run_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with (args.run_dir / 'loss.jsonl').open('a') as log:
        for step in range(start, args.steps):
            group = groups[keys[step % len(keys)]]
            row = group[int(torch.randint(len(group), (1,)))]
            target, gold = targets[row['cache_dir']]
            lr = 2e-5 * min(1, (step + 1) / 20) * (0.1 + 0.45 * (1 + math.cos(math.pi * step / 320)))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            optimizer.zero_grad(set_to_none=True)
            prefix, mask, lengths = reader(features[row['id']])
            length = int(lengths[0])
            if length != target.shape[1] or not bool(mask[0, :length].all()):
                raise ValueError('Teacher/reader constant prefix length differs')
            loss, metrics = acoustic_objective(wrapper, builder, prefix[:, :length], speaker, gold, target, reader.scale,
                                               prefix_weight=args.prefix_weight)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite acoustic loss; inspect before retry')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(reader.parameters(), 1.0)
            if not torch.isfinite(norm) or norm <= 0:
                raise RuntimeError('Reader gradients missing/nonfinite')
            if any(p.requires_grad or p.grad is not None for p in wrapper.model.parameters()):
                raise RuntimeError('Backbone must remain frozen with absent gradients')
            optimizer.step()
            metric = {'step': step + 1, 'loss': float(loss.detach()),
                      **{k: float(v.detach()) for k, v in metrics.items()},
                      'gradient_norm': float(norm), 'learning_rate': lr,
                      'elapsed_seconds': time.monotonic() - started, 'training_id': row['id']}
            log.write(json.dumps(metric) + '\n');log.flush()
            if step % 10 == 0 or step + 1 == args.steps:
                print(json.dumps(metric), flush=True)
            if (step + 1) % 40 == 0 or step + 1 == args.steps:
                info.update(peak_allocated_vram_bytes=(torch.cuda.max_memory_allocated()
                                                      if speaker.device.type == 'cuda' else None),
                            backbone_gradients_absent=True)
                save(args.run_dir, reader, optimizer, step + 1, info)


if __name__ == '__main__':
    main()
