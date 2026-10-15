"""Calibrate numeric image vectors through frozen continuous opening likelihood."""
import argparse
import json
import math
import os
from pathlib import Path

import torch
from evaluate_autoregressive_prior import load_prior
from safetensors.torch import load_file, save_file

from latent_diaries.inversion import digest
from latent_diaries.prior_calibration import NumericImageAdapter, opening_nll, shared_opening_length
from latent_diaries.visual_anchors import training_anchor_targets


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image-dataset', type=Path, required=True)
    p.add_argument('--labels', type=Path, required=True)
    p.add_argument('--corpus-dir', type=Path, required=True)
    p.add_argument('--prior-dir', type=Path, required=True)
    p.add_argument('--initial-adapter', type=Path, required=True)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--steps', type=int, required=True)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    if not Path('/.dockerenv').exists() or not 1 <= args.steps <= 1000:
        raise ValueError('Managed worker and prescribed1000-step schedule required')
    if args.run_dir.exists() and not args.resume:
        raise ValueError('Use fresh output or explicit resume')
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    initial = json.loads((args.initial_adapter.parent / 'manifest.json').read_text())
    prior_info = json.loads((args.prior_dir / 'config.json').read_text())
    prior_file = args.prior_dir / 'reader.safetensors'
    corpus_file = args.corpus_dir / 'corpus.json'
    prefix_file = args.corpus_dir / 'prefixes.safetensors'
    corpus = json.loads(corpus_file.read_text())
    if (digest(args.initial_adapter) != initial['adapter_sha256'] or
            digest(prior_file) != prior_info['reader_sha256'] or
            digest(corpus_file) != initial['corpus_sha256'] or
            digest(prefix_file) != corpus['provenance']['prefix_sha256'] or
            digest(args.image_dataset) != initial['image_dataset_sha256'] or
            digest(args.labels) != initial['anchor_labels_sha256'] or
            digest(Path('uv.lock')) != prior_info['lock_sha256'] or
            any(initial[k] != prior_info[k] for k in ['corpus_sha256', 'features_sha256'])):
        raise ValueError('Frozen initialization/cache pairing differs')
    rows = [r for r in json.loads(args.image_dataset.read_text())['rows'] if r['split'] == 'train']
    labels = [json.loads(s) for s in args.labels.read_text().splitlines() if s]
    targets = training_anchor_targets(labels, {r['id'] for r in rows})
    rows = [r for r in rows if targets.get(r['id']) is not None]
    image = torch.tensor([r['features'] for r in rows], device=device)
    by_anchor = {}
    for i, row in enumerate(rows):
        by_anchor.setdefault(targets[row['id']], []).append(i)
    keys = sorted(by_anchor)
    if len(keys) < 2:
        raise ValueError('At least two visible training anchors required')
    by_text = {key: [i for i, r in enumerate(corpus['rows']) if r['split'] == 'train'
                    and f"{r['category']}:{r['opening_index']}" == key] for key in keys}
    prefix = load_file(str(prefix_file))['prefixes'].to(device)
    lengths = torch.tensor([r['length'] for r in corpus['rows']], device=device)
    boundaries = {key: shared_opening_length(prefix[ids], lengths[ids]) for key, ids in by_text.items()}
    maximum = max(boundaries.values())
    prior = load_prior(load_file(str(prior_file)), prior_info['recipe'], prior_info['modes']).to(device).eval().requires_grad_(False)
    starting = load_file(str(args.initial_adapter))['coefficient'].to(device)
    adapter = NumericImageAdapter(starting).to(device)
    with torch.no_grad():
        reference = adapter(image).detach()
        null = image.new_zeros(1, 768)
        null_reference = adapter(null).detach()
    info = {k: initial[k] for k in ['corpus_sha256', 'features_sha256', 'image_dataset_sha256',
                                   'design_sha256', 'anchor_labels_sha256', 'supported_categories']}
    info.update(scope='Prior-aware numeric calibration; existing training images only',
                initial_adapter_sha256=digest(args.initial_adapter), initial_manifest_sha256=digest(args.initial_adapter.parent / 'manifest.json'),
                prior_sha256=digest(prior_file), prior_config_sha256=digest(args.prior_dir / 'config.json'),
                prefix_sha256=digest(prefix_file), lock_sha256=digest(Path('uv.lock')),
                protocol_sha256=digest(Path('configs/research/prior-aware-adapter-v12.json')),
                source_sha256=digest(Path(__file__)), loss_source_sha256=digest(Path('src/latent_diaries/prior_calibration.py')),
                training_cases=len(rows), training_cases_excluded=initial['training_cases_excluded'],
                trainable_parameters=starting.numel(), frozen_prior_parameters=sum(v.numel() for v in prior.parameters()),
                seed=42, learning_rate=5e-5, schedule_steps=1000, source_revision=os.environ.get('LD_CODE_REVISION'))
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=5e-5, weight_decay=0)
    start = 0
    if args.resume:
        saved = json.loads((args.run_dir / 'manifest.json').read_text())
        if any(saved[k] != v for k, v in info.items() if k != 'source_revision'):
            raise ValueError('Resume provenance differs')
        if digest(args.run_dir / 'adapter.safetensors') != saved['adapter_sha256']:
            raise ValueError('Saved adapter changed')
        adapter.load_state_dict(load_file(str(args.run_dir / 'adapter.safetensors'), device=str(device)))
        state = torch.load(args.run_dir / 'state.pt', weights_only=True, map_location=device)
        if state['adapter_sha256'] != saved['adapter_sha256'] or state['step'] != saved['completed_steps']:
            raise ValueError('Checkpoint pair differs')
        optimizer.load_state_dict(state['optimizer']);start = state['step'];info = saved
        torch.set_rng_state(state['cpu_rng'].cpu())
        if state['cuda_rng']:
            torch.cuda.set_rng_state_all([r.cpu() for r in state['cuda_rng']])
    args.run_dir.mkdir(parents=True, exist_ok=True)

    def choose(values):
        return values[int(torch.randint(len(values), (1,)))]

    def diagnostic():
        ids = [by_anchor[k][0] for k in keys]
        texts = [by_text[k][0] for k in keys]
        lens = torch.tensor([boundaries[k] for k in keys], device=device)
        with torch.no_grad():
            semantic = adapter(image[ids])
            gold = prefix[texts, :maximum]
            positive = opening_nll(prior, semantic, gold, lens)
            negative = opening_nll(prior, semantic.roll(1, 0), gold, lens)
            return {'cases': len(keys), 'matched_nll': float(positive.mean()), 'shuffled_nll': float(negative.mean()),
                    'semantic_drift_squared': float((semantic - reference[ids]).square().sum(-1).mean())}

    initial_diagnostic = diagnostic()
    print(json.dumps({'start_step': start, 'diagnostic': initial_diagnostic, 'trainable_parameters': starting.numel()}), flush=True)
    with (args.run_dir / 'loss.jsonl').open('a') as log:
        for step in range(start, args.steps):
            chosen_keys = [choose(keys) for _ in range(16)]
            ids = [choose(by_anchor[k]) for k in chosen_keys]
            texts = [choose(by_text[k]) for k in chosen_keys]
            negative_keys = []
            for key in chosen_keys:
                candidates = [k for k in keys if k != key and k.split(':')[0] == key.split(':')[0]]
                negative_keys.append(choose(candidates or [k for k in keys if k != key]))
            negative_ids = [choose(by_anchor[k]) for k in negative_keys]
            lens = torch.tensor([boundaries[k] for k in chosen_keys], device=device)
            gold = prefix[texts, :maximum]
            lr = 5e-5 * min(1, (step + 1) / 50) * (0.1 + 0.45 * (1 + math.cos(math.pi * step / 1000)))
            for group in optimizer.param_groups:
                group['lr'] = lr
            optimizer.zero_grad(set_to_none=True)
            semantic = adapter(image[ids])
            positive = opening_nll(prior, semantic, gold, lens)
            negative = opening_nll(prior, adapter(image[negative_ids]), gold, lens)
            ranking = torch.relu(0.2 + positive - negative).mean()
            drift = (semantic - reference[ids]).square().sum(-1).mean() + (adapter(null) - null_reference).square().sum()
            loss = positive.mean() + 0.5 * ranking + 20 * drift
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1)
            if not torch.isfinite(loss) or not torch.isfinite(norm) or norm <= 0 or any(v.grad is not None for v in prior.parameters()):
                raise RuntimeError('Invalid calibration gradient or frozen-prior mutation')
            optimizer.step()
            record = {'step': step + 1, 'loss': float(loss.detach()), 'matched_nll': float(positive.detach().mean()),
                      'negative_nll': float(negative.detach().mean()), 'ranking': float(ranking.detach()),
                      'drift_squared': float(drift.detach()), 'gradient_norm': float(norm), 'learning_rate': lr}
            log.write(json.dumps(record) + '\n');log.flush()
            if (step + 1) % 100 == 0 or step == start:
                print(json.dumps(record), flush=True)
            if step + 1 in {500, 1000}:
                file = args.run_dir / 'adapter.tmp.safetensors'
                save_file({'coefficient': adapter.coefficient.detach().cpu().contiguous()}, str(file))
                checksum = digest(file)
                torch.save({'step': step + 1, 'adapter_sha256': checksum, 'optimizer': optimizer.state_dict(),
                            'cpu_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else []},
                           args.run_dir / 'state.tmp.pt')
                file.replace(args.run_dir / 'adapter.safetensors')
                (args.run_dir / 'state.tmp.pt').replace(args.run_dir / 'state.pt')
                info.update(completed_steps=step + 1, adapter_sha256=checksum,
                            peak_allocated_vram_bytes=torch.cuda.max_memory_allocated() if device.type == 'cuda' else None,
                            opening_diagnostic=diagnostic())
                (args.run_dir / 'manifest.tmp.json').write_text(json.dumps(info, indent=2) + '\n')
                (args.run_dir / 'manifest.tmp.json').replace(args.run_dir / 'manifest.json')
    if digest(prior_file) != info['prior_sha256']:
        raise RuntimeError('Frozen prior file changed')


if __name__ == '__main__':
    main()
