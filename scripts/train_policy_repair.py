"""Repair closed-loop policy drift using mixed gold and student-generated histories."""
import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
from image_probe import save
from safetensors.torch import load_file, save_file

from latent_diaries.anchored_reader import initialization_fingerprint, make_anchored_reader
from latent_diaries.inversion import digest, distribution_kl, load_backbone, trajectory_logits
from latent_diaries.talker_lora import inject_talker_lora, load_lora_state, lora_parameters, lora_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--rollout-cache', type=Path, required=True)
    parser.add_argument('--policy-cache', type=Path, required=True)
    parser.add_argument('--teachers', type=Path, required=True)
    parser.add_argument('--anchor-run', type=Path, required=True)
    parser.add_argument('--provenance-run', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--steps', type=int, required=True)
    parser.add_argument('--unit-scale', action='store_true')
    parser.add_argument('--head-lr-multiplier', type=float, choices=[5.], default=5.)
    parser.add_argument('--lora-rank', type=int, choices=[0], default=0)
    parser.add_argument('--lora-layers', type=int, choices=[4], default=4)
    parser.add_argument('--schedule-steps', type=int, default=320)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if not Path('/.dockerenv').exists() or not 1 <= args.steps <= args.schedule_steps <= 3200:
        raise ValueError('Use managed worker, within a fixed schedule of at most3200 steps')
    if args.run_dir.exists() and not args.resume:
        raise ValueError('Use a fresh run directory or explicit resume')
    info = json.loads((args.provenance_run / 'config.json').read_text())
    initial_path = args.provenance_run / 'reader.safetensors'
    if digest(initial_path) != info['reader_sha256'] or digest(args.dataset) != info['dataset_sha256']:
        raise ValueError('Initial model/dataset fingerprint differs')
    if info['evaluation_scope'] != 'development':
        raise ValueError('Expected constant-length development initialization')
    anchor_info = json.loads((args.anchor_run / 'config.json').read_text())
    for key in ['model_id', 'model_revision', 'lock_sha256']:
        if anchor_info[key] != info[key]:
            raise ValueError('Anchor backbone/dependency provenance differs')
    anchor_path = args.anchor_run / 'prefix.safetensors'
    anchor = load_file(str(anchor_path))['soft_tokens'][0]
    info.pop('training_prefix_sha256', None)
    info.pop('trainable_parameters', None)
    info.pop('initialization_sha256', None)
    info.update(maximum_tokens=len(anchor), fixed_length=True, width=256,
                architecture='global-mlp', anchor_sha256=digest(anchor_path),
                anchor_config_sha256=digest(args.anchor_run / 'config.json'),
                anchored_source_sha256=digest(Path('src/latent_diaries/anchored_reader.py')),
                prefix_regression_initialization_used=False, training_text_prefix_targets_loaded=False)
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
    policies_path = args.policy_cache / 'manifest.json'
    policies = json.loads(policies_path.read_text())
    if policies['teacher_manifest_sha256'] != digest(teacher_manifest) or policies['lock_sha256'] != digest(Path('uv.lock')):
        raise ValueError('Teacher policy provenance differs')
    policy_by_key = {r['prefix_key']:r for r in policies['records']}
    if set(policy_by_key) != {str(Path(r['prefix_file']).parent) for r in teachers['records']}:
        raise ValueError('Teacher policy coverage differs')
    if len(policy_by_key) != len(policies['records']):
        raise ValueError('Duplicate teacher policy keys')
    rollout_path = args.rollout_cache / 'manifest.json'
    rollouts = json.loads(rollout_path.read_text())
    if (rollouts['dataset_sha256'] != digest(args.dataset)
            or rollouts['teacher_manifest_sha256'] != digest(teacher_manifest)
            or rollouts['source_reader_sha256'] != digest(initial_path)
            or rollouts['source_reader_config_sha256'] != digest(args.provenance_run/'config.json')):
        raise ValueError('Rollout bank does not belong to the declared warm-start reader/data')
    info.update(rollout_manifest_sha256=digest(rollout_path), warm_start_reader_sha256=digest(initial_path),
                history_gold_probability=0.5, unit_scale=args.unit_scale, head_lr_multiplier=args.head_lr_multiplier,
                policy_manifest_sha256=digest(policies_path), provenance_reader_sha256=digest(initial_path), completed_steps=0,
                teacher_manifest_sha256=digest(teacher_manifest),
                provenance_config_sha256=digest(args.provenance_run / 'config.json'),
                source_sha256=digest(Path(__file__)),
                acoustic_source_sha256=digest(Path('src/latent_diaries/acoustic_reader.py')),
                conditioning_source_sha256=digest(Path('src/latent_diaries/conditioning.py')),
                inversion_source_sha256=digest(Path('src/latent_diaries/inversion.py')),
                source_revision=os.environ.get('LD_CODE_REVISION'), learning_rate=2e-5,
                schedule_steps=args.schedule_steps, warmup_steps=20,
                prefix_weight=0.0,
                objective='codec0 KL + residual KL on mixed gold/student-generated audio histories; no prefix target',
                scope='Frozen-Talker direct visual policy repair; equal gold/generated history sampling',
                talker_lora_rank=args.lora_rank, talker_lora_layers=args.lora_layers,
                lora_learning_rate_multiplier=1.0, qwen_weights_changed=bool(args.lora_rank),
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
    modules = (inject_talker_lora(wrapper.model.talker, rank=args.lora_rank, layers=args.lora_layers)
               if args.lora_rank else [])
    adapted = lora_parameters(wrapper.model.talker)
    allowed = {id(p) for p in adapted}
    if args.resume and info['lora_modules'] != modules:
        raise ValueError('LoRA target module layout differs')
    info.update(lora_modules=modules, lora_trainable_parameters=sum(p.numel() for p in adapted))
    torch.manual_seed(42)
    reader = make_anchored_reader(anchor, width=256, unit_scale=args.unit_scale).to(speaker.device).train()
    reader.load_state_dict(load_file(str(initial_path)))
    initial_checksum = initialization_fingerprint(reader)
    if args.resume and info['initialization_sha256'] != initial_checksum:
        raise ValueError('Direct reader initialization differs')
    info.update(initialization_sha256=initial_checksum,
                trainable_parameters=sum(p.numel() for p in reader.parameters()))
    head = list(reader.network[-1].parameters())
    head_ids = {id(p) for p in head}
    other = [p for p in reader.parameters() if id(p) not in head_ids]
    optimizer = torch.optim.AdamW([{'params': other, 'lr_multiplier': 1.0},
                                   {'params': head, 'lr_multiplier': args.head_lr_multiplier},
                                   {'params': adapted, 'lr_multiplier': 1.0}], lr=2e-5, weight_decay=0)
    targets = {}
    for record in teachers['records']:
        if digest(Path(record['codes_file'])) != record['codes_sha256']:
            raise ValueError('Teacher audio tensor changed')
        gold = load_file(record['codes_file'])['audio_codes'].to(speaker.device)
        if gold.ndim != 3 or gold.shape[0] != 1 or gold.shape[-1] != 16 or gold.dtype != torch.long:
            raise ValueError('Expected gold [1,F,16] integer codes')
        # The path is an opaque training index only. No text prefix tensor is loaded.
        key = str(Path(record['prefix_file']).parent)
        policy = policy_by_key[key]
        if policy['codes_sha256'] != record['codes_sha256'] or digest(Path(policy['file'])) != policy['sha256']:
            raise ValueError('Teacher policy/code pairing differs')
        values = load_file(policy['file'])
        first, rest = values['codec0_logits'], values['residual_logits']
        if first.shape[1] != gold.shape[1]+1 or rest.shape[:2] != (gold.shape[1], 15):
            raise ValueError('Teacher policy alignment differs')
        targets[key] = (gold, first.to(speaker.device), rest.to(speaker.device))
    rollout_targets = {}
    for record in rollouts['records']:
        if digest(Path(record['file'])) != record['sha256']:
            raise ValueError('Student history cache changed')
        values = load_file(record['file'])
        codes = values['audio_codes']
        first, rest = values['codec0_logits'], values['residual_logits']
        if codes.shape[0] != 1 or codes.shape[-1] != 16 or codes.dtype != torch.long:
            raise ValueError('Expected student [1,F,16] codes')
        if first.shape[1] != codes.shape[1]+1 or rest.shape[:2] != (codes.shape[1], 15):
            raise ValueError('Student history target alignment differs')
        rollout_targets[record['prefix_key']] = tuple(v.to(speaker.device) for v in [codes, first, rest])
    if not rollout_targets or not set(rollout_targets) <= set(targets):
        raise ValueError('Student histories must be a nonempty subset of training targets')
    groups = {key: [r for r in training if r['cache_dir'] == key] for key in sorted(targets)}
    if any(not group for group in groups.values()) or sum(map(len, groups.values())) != len(training):
        raise ValueError('Teacher/class training coverage differs')
    features = {r['id']: torch.tensor([r['features']], dtype=torch.float32, device=speaker.device)
                for r in training}
    keys = sorted(groups)
    rollout_keys = sorted(rollout_targets)
    start = 0
    torch.manual_seed(42)
    if args.resume:
        if digest(args.run_dir / 'reader.safetensors') != info['reader_sha256']:
            raise ValueError('Reader checksum differs')
        reader.load_state_dict(load_file(str(args.run_dir / 'reader.safetensors')))
        if args.lora_rank:
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
            use_rollout = bool(torch.rand(()) >= 0.5)
            available = rollout_keys if use_rollout else keys
            key = available[int(torch.randint(len(available), (1,)))]
            group = groups[key]
            row = group[int(torch.randint(len(group), (1,)))]
            history, teacher0, teacher_rest = (rollout_targets if use_rollout else targets)[key]
            lr = 2e-5 * min(1, (step + 1) / 20) * (0.1 + 0.45 * (1 + math.cos(math.pi * step / args.schedule_steps)))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr * param_group['lr_multiplier']
            optimizer.zero_grad(set_to_none=True)
            prefix, mask, lengths = reader(features[row['id']])
            length = int(lengths[0])
            if length != info['maximum_tokens'] or not bool(mask.all()):
                raise ValueError('Fixed visual prefix protocol differs')
            student0, student_rest = trajectory_logits(wrapper, builder, prefix[:, :length], speaker, history)
            kl0 = distribution_kl(student0, teacher0)
            kl_rest = distribution_kl(student_rest, teacher_rest)
            loss = kl0 + kl_rest
            metrics = {'codec0_kl':kl0, 'residual_kl':kl_rest}
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite acoustic loss; inspect before retry')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(list(reader.parameters()) + adapted, 1.0)
            if not torch.isfinite(norm) or norm <= 0:
                raise RuntimeError('Reader gradients missing/nonfinite')
            if any((p.requires_grad or p.grad is not None) and id(p) not in allowed
                   for p in wrapper.model.parameters()):
                raise RuntimeError('Vendor backbone must remain frozen; only declared LoRA may receive gradients')
            if adapted and not any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in adapted):
                raise RuntimeError('LoRA gradients missing')
            optimizer.step()
            metric = {'step': step + 1, 'loss': float(loss.detach()),
                      **{k: float(v.detach()) for k, v in metrics.items()},
                      'gradient_norm': float(norm), 'learning_rate': lr, 'head_learning_rate': lr * args.head_lr_multiplier, 'lora_learning_rate': lr if adapted else 0.0,
                      'elapsed_seconds': time.monotonic() - started, 'training_id': row['id'],
                      'history_source': 'student_generated' if use_rollout else 'gold'}
            log.write(json.dumps(metric) + '\n');log.flush()
            if step % 10 == 0 or step + 1 == args.steps:
                print(json.dumps(metric), flush=True)
            if (step + 1) % 40 == 0 or step + 1 == args.steps:
                info.update(peak_allocated_vram_bytes=(torch.cuda.max_memory_allocated()
                                                      if speaker.device.type == 'cuda' else None),
                            backbone_gradients_absent=not bool(adapted), vendor_base_gradients_absent=True)
                if adapted:
                    save_file(lora_state(wrapper.model.talker), str(args.run_dir / 'lora.tmp.safetensors'))
                    info['lora_sha256'] = digest(args.run_dir / 'lora.tmp.safetensors')
                    (args.run_dir / 'lora.tmp.safetensors').replace(args.run_dir / 'lora.safetensors')
                save(args.run_dir, reader, optimizer, step + 1, info)


if __name__ == '__main__':
    main()
