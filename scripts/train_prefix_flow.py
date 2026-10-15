"""Train continuous language prior in one managed worker, with actual resume."""
import argparse
import json
import math
import os
from pathlib import Path

import torch
from image_probe import save
from safetensors.torch import load_file

from latent_diaries.inversion import digest
from latent_diaries.prefix_flow import PrefixFlow, flow_loss


def make_flow(weights, width):
    model = PrefixFlow(*(weights[k] for k in ['mean', 'scale', 'center', 'basis', 'latent_scale']), width=width)
    model.load_state_dict(weights)
    return model


def fit_subspace(prefix, lengths):
    mask = torch.arange(prefix.shape[1], device=prefix.device)[None] < lengths[:, None]
    mean = prefix[mask].mean(0)
    scale = prefix[mask].std(0, unbiased=False).clamp_min(1e-4)
    normalized = ((prefix - mean) / scale).flatten(1)
    center = normalized.mean(0)
    matrix = (normalized - center).double()
    eigenvalues, vectors = torch.linalg.eigh(matrix @ matrix.T)
    order = torch.arange(len(eigenvalues) - 1, -1, -1, device=matrix.device)
    eigenvalues, vectors = eigenvalues[order], vectors[:, order]
    ranks = sorted({r for r in [128, 256, 512, 768, 1024, len(prefix) - 1] if r < len(prefix)})
    floors = {}
    for rank in ranks:
        basis = (vectors[:, :rank].T @ matrix / eigenvalues[:rank].sqrt()[:, None]).float()
        coordinates = (normalized - center) @ basis.T
        reconstruction = ((coordinates @ basis + center).reshape_as(prefix) * scale + mean)
        error = float((reconstruction - prefix).square().mean())
        floors[rank] = error
        if error <= 1e-7:
            latent_scale = coordinates.std(0, unbiased=False).clamp_min(1e-4)
            return (mean, scale, center, basis, latent_scale), {'rank': rank, 'training_raw_mse': error,
                                                              'candidate_floors': floors}
    raise RuntimeError('Training-only subspace floor fails; inspect before fitting')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus-dir', type=Path, required=True)
    parser.add_argument('--features-dir', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--steps', type=int, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if not Path('/.dockerenv').exists() or not 1 <= args.steps <= 2400:
        raise ValueError('Use managed worker within prescribed2400-step schedule')
    if args.run_dir.exists() and not args.resume:
        raise ValueError('Use a fresh run directory or explicit resume')
    corpus_file = args.corpus_dir / 'corpus.json'
    corpus = json.loads(corpus_file.read_text())
    feature_manifest = json.loads((args.features_dir / 'manifest.json').read_text())
    prefix_file, feature_file = args.corpus_dir / 'prefixes.safetensors', args.features_dir / 'features.safetensors'
    if (feature_manifest['corpus_sha256'] != digest(corpus_file) or
            feature_manifest['features_sha256'] != digest(feature_file) or
            corpus['provenance']['prefix_sha256'] != digest(prefix_file)):
        raise ValueError('Cached training tensors changed')
    if feature_manifest['ids'] != [r['id'] for r in corpus['rows']]:
        raise ValueError('Text vector/prefix row boundaries differ')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    prefix = load_file(str(prefix_file))['prefixes'].to(device)
    features = load_file(str(feature_file))['embeddings'].to(device)
    lengths = torch.tensor([r['length'] for r in corpus['rows']], device=device)
    train_mask = torch.tensor([r['split'] == 'train' for r in corpus['rows']], device=device)
    info = {'corpus_sha256': digest(corpus_file), 'features_sha256': digest(feature_file),
            'prefix_sha256': digest(prefix_file), 'source_sha256': digest(Path(__file__)),
            'reader_source_sha256': digest(Path('src/latent_diaries/prefix_flow.py')),
            'lock_sha256': digest(Path('uv.lock')), 'protocol_sha256': digest(Path('configs/research/semantic-prefix-v5.json')),
            'schedule_steps': 2400, 'warmup_steps': 100, 'learning_rate': 2e-4, 'seed': 42,
            'source_revision': os.environ.get('LD_CODE_REVISION'), 'scope': 'Training-only text-feature language prior',
            **{k: corpus['provenance'][k] for k in ['model_id', 'model_revision']}}
    torch.manual_seed(42)
    if args.resume:
        saved = json.loads((args.run_dir / 'config.json').read_text())
        for key, value in info.items():
            if key != 'source_revision' and saved[key] != value:
                raise ValueError(f'Resume provenance differs: {key}')
        info = saved
        if digest(args.run_dir / 'reader.safetensors') != info['reader_sha256']:
            raise ValueError('Reader checksum changed')
        model = make_flow(load_file(str(args.run_dir / 'reader.safetensors')), info['width']).to(device)
    else:
        buffers, subspace = fit_subspace(prefix[train_mask], lengths[train_mask])
        width = max(512, math.ceil((subspace['rank'] + 1) / 128) * 128)
        model = PrefixFlow(*buffers, width=width).to(device)
        info.update(width=width, subspace=subspace, trainable_parameters=sum(p.numel() for p in model.parameters()))
        if info['trainable_parameters'] > 25_000_000:
            raise ValueError('Reader exceeds compact-model budget')
        print(json.dumps({'subspace': subspace, 'parameters': info['trainable_parameters']}), flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0)
    start = 0
    if args.resume:
        state = torch.load(args.run_dir / 'state.pt', weights_only=True, map_location=device)
        if state['reader_sha256'] != info['reader_sha256'] or state['step'] != info['completed_steps']:
            raise ValueError('Checkpoint pair inconsistent')
        optimizer.load_state_dict(state['optimizer']);start = state['step']
        torch.set_rng_state(state['cpu_rng'].cpu())
        if state['cuda_rng']:
            torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda_rng']])
    clean = model.pack(prefix, lengths).detach()
    training = train_mask.nonzero()[:, 0]
    args.run_dir.mkdir(parents=True, exist_ok=True)
    with (args.run_dir / 'loss.jsonl').open('a') as log:
        for step in range(start, args.steps):
            chosen = training[torch.randint(len(training), (32,), device=device)]
            gold = clean[chosen]
            noise = torch.randn_like(gold)
            time = torch.rand(32, device=device)
            interpolated = (1 - time[:, None]) * noise + time[:, None] * gold
            lr = 2e-4 * min(1, (step + 1) / 100) * (0.1 + 0.45 * (1 + math.cos(math.pi * step / 2400)))
            for group in optimizer.param_groups:
                group['lr'] = lr
            optimizer.zero_grad(set_to_none=True)
            loss = flow_loss(model.velocity(interpolated, features[chosen], time), gold - noise)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite flow loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            if not torch.isfinite(norm) or norm <= 0:
                raise RuntimeError('Invalid flow gradients')
            optimizer.step()
            metric = {'step': step + 1, 'velocity_loss': float(loss.detach()), 'gradient_norm': float(norm), 'learning_rate': lr}
            log.write(json.dumps(metric) + '\n');log.flush()
            if (step + 1) % 100 == 0 or step == start:
                print(json.dumps(metric), flush=True)
            if (step + 1) % 400 == 0 or step + 1 == args.steps:
                info['peak_allocated_vram_bytes'] = torch.cuda.max_memory_allocated() if device.type == 'cuda' else None
                save(args.run_dir, model, optimizer, step + 1, info)


if __name__ == '__main__':
    main()
