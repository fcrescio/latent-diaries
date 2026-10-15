"""Fit causal next-vector prior on cached training-only text supervision."""
import argparse
import json
import math
import os
from pathlib import Path

import torch
from image_probe import save
from safetensors.torch import load_file

from latent_diaries.autoregressive_prior import AutoregressivePrior, causal_loss
from latent_diaries.inversion import digest


def make_prior(weights):
    model = AutoregressivePrior(*(weights[k] for k in ['mean', 'scale', 'basis', 'latent_scale']))
    model.load_state_dict(weights)
    return model


def token_subspace(prefix, lengths):
    valid = torch.arange(prefix.shape[1], device=prefix.device)[None] < lengths[:, None]
    values = prefix[valid]
    mean, scale = values.mean(0), values.std(0, unbiased=False).clamp_min(1e-4)
    x = ((values - mean) / scale).double()
    eigenvalues, eigenvectors = torch.linalg.eigh(x.T @ x)
    eigenvectors = eigenvectors.flip(-1)
    floors = {}
    for rank in [64, 128, 192, 256, 320, 384, 512, 768, 1024]:
        basis = eigenvectors[:, :rank].T.float()
        coordinate = x.float() @ basis.T
        recovered = coordinate @ basis * scale + mean
        error = float((recovered - values).square().mean())
        floors[rank] = error
        if error <= 1e-7:
            latent_scale = coordinate.std(0, unbiased=False).clamp_min(0.1)
            return (mean, scale, basis, latent_scale), {'rank': rank, 'training_raw_mse': error,
                                                       'candidate_floors': floors}
    raise RuntimeError('Token subspace failed full-rank reconstruction')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus-dir', type=Path, required=True)
    parser.add_argument('--features-dir', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--steps', type=int, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if not Path('/.dockerenv').exists() or not 1 <= args.steps <= 4000:
        raise ValueError('Use managed worker within prescribed4000-step schedule')
    if args.run_dir.exists() and not args.resume:
        raise ValueError('Use fresh run directory or explicit resume')
    corpus_file = args.corpus_dir / 'corpus.json'
    corpus = json.loads(corpus_file.read_text())
    manifest = json.loads((args.features_dir / 'manifest.json').read_text())
    prefix_file, feature_file = args.corpus_dir / 'prefixes.safetensors', args.features_dir / 'features.safetensors'
    if (manifest['corpus_sha256'] != digest(corpus_file) or manifest['features_sha256'] != digest(feature_file)
            or corpus['provenance']['prefix_sha256'] != digest(prefix_file)):
        raise ValueError('Cached corpus/features/targets changed')
    if manifest['ids'] != [r['id'] for r in corpus['rows']]:
        raise ValueError('Cached row alignment differs')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    prefix = load_file(str(prefix_file))['prefixes'].to(device)
    features = load_file(str(feature_file))['embeddings'].to(device)
    lengths = torch.tensor([r['length'] for r in corpus['rows']], device=device)
    training_mask = torch.tensor([r['split'] == 'train' for r in corpus['rows']], device=device)
    info = {'corpus_sha256': digest(corpus_file), 'features_sha256': digest(feature_file),
            'prefix_sha256': digest(prefix_file), 'source_sha256': digest(Path(__file__)),
            'reader_source_sha256': digest(Path('src/latent_diaries/autoregressive_prior.py')),
            'lock_sha256': digest(Path('uv.lock')), 'protocol_sha256': digest(Path('configs/research/autoregressive-prefix-v6.json')),
            'schedule_steps': 4000, 'warmup_steps': 100, 'learning_rate': 3e-4, 'seed': 42,
            'source_revision': os.environ.get('LD_CODE_REVISION'), 'scope': 'Causal continuous language prior; training-text supervision',
            **{k: corpus['provenance'][k] for k in ['model_id', 'model_revision']}}
    torch.manual_seed(42)
    if args.resume:
        saved = json.loads((args.run_dir / 'config.json').read_text())
        for key, value in info.items():
            if key != 'source_revision' and value != saved[key]:
                raise ValueError(f'Resume provenance differs: {key}')
        info = saved
        if digest(args.run_dir / 'reader.safetensors') != info['reader_sha256']:
            raise ValueError('Reader checksum differs')
        model = make_prior(load_file(str(args.run_dir / 'reader.safetensors'))).to(device)
    else:
        buffers, subspace = token_subspace(prefix[training_mask], lengths[training_mask])
        model = AutoregressivePrior(*buffers).to(device)
        info.update(subspace=subspace, trainable_parameters=sum(p.numel() for p in model.parameters()))
        if info['trainable_parameters'] > 25_000_000:
            raise ValueError('Reader exceeds compact model budget')
        print(json.dumps({'subspace': subspace, 'parameters': info['trainable_parameters']}), flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0)
    start = 0
    if args.resume:
        state = torch.load(args.run_dir / 'state.pt', weights_only=True, map_location=device)
        if state['reader_sha256'] != info['reader_sha256'] or state['step'] != info['completed_steps']:
            raise ValueError('Checkpoint pair inconsistent')
        optimizer.load_state_dict(state['optimizer']);start = state['step']
        torch.set_rng_state(state['cpu_rng'].cpu())
        if state['cuda_rng']:
            torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda_rng']])
    training = training_mask.nonzero()[:, 0]
    args.run_dir.mkdir(parents=True, exist_ok=True)
    with (args.run_dir / 'loss.jsonl').open('a') as log:
        for step in range(start, args.steps):
            chosen = training[torch.randint(len(training), (16,), device=device)]
            lr = 3e-4 * min(1, (step + 1) / 100) * (0.1 + 0.45 * (1 + math.cos(math.pi * step / 4000)))
            for group in optimizer.param_groups:
                group['lr'] = lr
            optimizer.zero_grad(set_to_none=True)
            loss, components = causal_loss(model, features[chosen], prefix[chosen], lengths[chosen])
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite causal flow loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            if not torch.isfinite(norm) or norm <= 0:
                raise RuntimeError('Causal reader gradients missing/nonfinite')
            optimizer.step()
            record = {'step': step + 1, 'loss': float(loss.detach()),
                      **{k: float(v.detach()) for k, v in components.items()},
                      'gradient_norm': float(norm), 'learning_rate': lr}
            log.write(json.dumps(record) + '\n');log.flush()
            if (step + 1) % 100 == 0 or step == start:
                print(json.dumps(record), flush=True)
            if (step + 1) % 500 == 0 or step + 1 == args.steps:
                info['peak_allocated_vram_bytes'] = torch.cuda.max_memory_allocated() if device.type == 'cuda' else None
                save(args.run_dir, model, optimizer, step + 1, info)


if __name__ == '__main__':
    main()
