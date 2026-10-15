"""Bounded reader plus final-attention LoRA control; vendor parameters stay frozen."""
import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
from image_probe import make_reader, save
from safetensors.torch import load_file, save_file

from latent_diaries.acoustic_reader import acoustic_objective
from latent_diaries.inversion import digest, load_backbone
from latent_diaries.talker_lora import inject_talker_lora, load_lora_state, lora_parameters, lora_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--teachers', type=Path, required=True)
    parser.add_argument('--initial-run', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--steps', type=int, required=True)
    parser.add_argument('--lora-rank', type=int, choices=[8], default=8)
    parser.add_argument('--lora-layers', type=int, choices=[4], default=4)
    parser.add_argument('--schedule-steps', type=int, default=320)
    parser.add_argument('--prefix-weight', type=float, default=0.1)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if not Path('/.dockerenv').exists() or not 1 <= args.steps <= args.schedule_steps <= 1600:
        raise ValueError('Use managed worker, within a fixed schedule of at most1600 steps')
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
                schedule_steps=args.schedule_steps, warmup_steps=20,
                prefix_weight=args.prefix_weight,
                objective=f'codec0 CE + 0.1 residual CE + {args.prefix_weight} normalized prefix MSE',
                scope='Image reader plus LoRA on last4 Talker q/v projections; vendor tensors frozen',
                talker_lora_rank=args.lora_rank, talker_lora_layers=args.lora_layers,
                lora_learning_rate_multiplier=5.0, qwen_weights_changed=True,
                vendor_base_weights_changed=False,
                lora_source_sha256=digest(Path('src/latent_diaries/talker_lora.py')))
    if args.resume:
        saved = json.loads((args.run_dir / 'config.json').read_text())
        for key, value in info.items():
            if key not in {'completed_steps', 'reader_sha256', 'source_revision',
                           'peak_allocated_vram_bytes'} and saved[key] != value:
                raise ValueError(f'Resume configuration differs: {key}')
        info = saved
    wrapper, builder, speaker = load_backbone({**info, 'talker_lora_rank': 0})
    for key in ['model_id', 'model_revision']:
        if teachers[key] != info[key]:
            raise ValueError('Teacher backbone provenance differs')
    if list(builder.role_ids) != teachers['role_ids']:
        raise ValueError('Teacher/student role protocol differs')
    torch.manual_seed(42)
    modules = inject_talker_lora(wrapper.model.talker, rank=args.lora_rank, layers=args.lora_layers)
    adapted = lora_parameters(wrapper.model.talker)
    allowed = {id(p) for p in adapted}
    if args.resume and info['lora_modules'] != modules:
        raise ValueError('LoRA target module layout differs')
    info.update(lora_modules=modules, lora_trainable_parameters=sum(p.numel() for p in adapted))
    reader = make_reader(info, load_file(str(initial_path))).to(speaker.device).train()
    optimizer = torch.optim.AdamW([{'params': list(reader.parameters()), 'lr_multiplier': 1.0},
                                   {'params': adapted, 'lr_multiplier': 5.0}], lr=2e-5, weight_decay=0)
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
        if digest(args.run_dir / 'lora.safetensors') != info['lora_sha256']:
            raise ValueError('LoRA checksum differs')
        load_lora_state(wrapper.model.talker, load_file(str(args.run_dir / 'lora.safetensors')))
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
            lr = 2e-5 * min(1, (step + 1) / 20) * (0.1 + 0.45 * (1 + math.cos(math.pi * step / args.schedule_steps)))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr * param_group['lr_multiplier']
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
            norm = torch.nn.utils.clip_grad_norm_(list(reader.parameters()) + adapted, 1.0)
            if not torch.isfinite(norm) or norm <= 0:
                raise RuntimeError('Reader gradients missing/nonfinite')
            if any((p.requires_grad or p.grad is not None) and id(p) not in allowed
                   for p in wrapper.model.parameters()):
                raise RuntimeError('Vendor backbone must remain frozen; only declared LoRA may receive gradients')
            if not any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in adapted):
                raise RuntimeError('LoRA gradients missing')
            optimizer.step()
            metric = {'step': step + 1, 'loss': float(loss.detach()),
                      **{k: float(v.detach()) for k, v in metrics.items()},
                      'gradient_norm': float(norm), 'learning_rate': lr, 'lora_learning_rate': lr * 5.0,
                      'elapsed_seconds': time.monotonic() - started, 'training_id': row['id']}
            log.write(json.dumps(metric) + '\n');log.flush()
            if step % 10 == 0 or step + 1 == args.steps:
                print(json.dumps(metric), flush=True)
            if (step + 1) % 40 == 0 or step + 1 == args.steps:
                info.update(peak_allocated_vram_bytes=(torch.cuda.max_memory_allocated()
                                                      if speaker.device.type == 'cuda' else None),
                            backbone_gradients_absent=False, vendor_base_gradients_absent=True)
                save_file(lora_state(wrapper.model.talker), str(args.run_dir / 'lora.tmp.safetensors'))
                info['lora_sha256'] = digest(args.run_dir / 'lora.tmp.safetensors')
                (args.run_dir / 'lora.tmp.safetensors').replace(args.run_dir / 'lora.safetensors')
                save(args.run_dir, reader, optimizer, step + 1, info)


if __name__ == '__main__':
    main()
