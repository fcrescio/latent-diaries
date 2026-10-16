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
from latent_diaries.multihypothesis_prior import MultihypothesisPrior, multihypothesis_loss


def make_prior(weights, recipe='causal-flow', modes=8):
    reader = MultihypothesisPrior if recipe == 'multihypothesis' else AutoregressivePrior
    kwargs = {'modes': modes} if recipe == 'multihypothesis' else {}
    model = reader(*(weights[k] for k in ['mean', 'scale', 'basis', 'latent_scale']), **kwargs)
    model.load_state_dict(weights)
    return model


def token_subspace(prefix, lengths):
    valid = torch.arange(prefix.shape[1], device=prefix.device)[None] < lengths[:, None]
    values = prefix[valid]
    mean, scale = values.mean(0), values.std(0, unbiased=False).clamp_min(1e-4)
    x = ((values - mean) / scale).double()
    _, eigenvectors = torch.linalg.eigh(x.T @ x)
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
    parser.add_argument('--protocol', type=Path)
    parser.add_argument('--steps', type=int, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--recipe', choices=['causal-flow', 'multihypothesis'], default='causal-flow')
    parser.add_argument('--initial-run', type=Path)
    parser.add_argument('--modes', type=int, default=8)
    parser.add_argument('--robust-semantics', action='store_true')
    parser.add_argument('--opening-semantics', action='store_true')
    args = parser.parse_args()
    schedule_steps = 3000 if args.opening_semantics else (6000 if args.robust_semantics else 4000)
    if not Path('/.dockerenv').exists() or not 1 <= args.steps <= schedule_steps:
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
    multi = args.recipe == 'multihypothesis'
    if args.robust_semantics and (not multi or args.modes != 32):
        raise ValueError('Robust semantic control prescribes32 continuous candidates')
    if args.opening_semantics and not args.robust_semantics:
        raise ValueError('Opening robustness requires the existing robust32-mode recipe')
    learning_rate = 1e-4 if args.opening_semantics else (3e-4 if args.robust_semantics else (5e-4 if multi else 3e-4))
    info = {'corpus_sha256': digest(corpus_file), 'features_sha256': digest(feature_file),
            'prefix_sha256': digest(prefix_file), 'source_sha256': digest(Path(__file__)),
            'reader_source_sha256': digest(Path('src/latent_diaries/autoregressive_prior.py')),
            'lock_sha256': digest(Path('uv.lock')), 'protocol_sha256': digest(Path('configs/research/autoregressive-prefix-v6.json')),
            'schedule_steps': schedule_steps, 'warmup_steps': 100, 'learning_rate': learning_rate, 'seed': 42,
            'recipe': args.recipe,
            'modes': args.modes, 'robust_semantics': args.robust_semantics,
            'opening_semantics': args.opening_semantics,
            'source_revision': os.environ.get('LD_CODE_REVISION'), 'scope': 'Causal continuous language prior; training-text supervision',
            **{k: corpus['provenance'][k] for k in ['model_id', 'model_revision']}}
    if multi:
        if not args.initial_run:
            raise ValueError('Multi-hypothesis recipe requires frozen causal initialization')
        initial_info = json.loads((args.initial_run / 'config.json').read_text())
        initial_file = args.initial_run / 'reader.safetensors'
        if digest(initial_file) != initial_info['reader_sha256']:
            raise ValueError('Initialization checksum differs')
        for key in ['corpus_sha256', 'features_sha256', 'prefix_sha256', 'lock_sha256']:
            if info[key] != initial_info[key]:
                raise ValueError('Initialization corpus/provenance differs')
        info.update(initial_reader_sha256=digest(initial_file), initial_config_sha256=digest(args.initial_run / 'config.json'),
                    hypothesis_source_sha256=digest(Path('src/latent_diaries/multihypothesis_prior.py')),
                    protocol_sha256=digest(Path('configs/research/multihypothesis-prefix-v7.json')))
        if args.robust_semantics:
            info['protocol_sha256'] = digest(Path('configs/research/robust-semantic-prior-v8.json'))
        if args.opening_semantics:
            if initial_info.get('modes') != 32 or not initial_info.get('robust_semantics'):
                raise ValueError('Opening control requires the completed robust32-mode initialization')
            info['protocol_sha256'] = digest(Path('configs/research/opening-robust-prior-v14.json'))
    if args.protocol:
        info['protocol_sha256'] = digest(args.protocol)
    torch.manual_seed(42)
    if args.resume:
        saved = json.loads((args.run_dir / 'config.json').read_text())
        for key, value in info.items():
            if key != 'source_revision' and value != saved[key]:
                raise ValueError(f'Resume provenance differs: {key}')
        info = saved
        if digest(args.run_dir / 'reader.safetensors') != info['reader_sha256']:
            raise ValueError('Reader checksum differs')
        model = make_prior(load_file(str(args.run_dir / 'reader.safetensors')), args.recipe, args.modes).to(device)
    else:
        if multi:
            initial_weights = load_file(str(initial_file))
            buffers = [initial_weights[k] for k in ['mean', 'scale', 'basis', 'latent_scale']]
            subspace = initial_info['subspace']
            model = MultihypothesisPrior(*buffers, modes=args.modes)
            loaded = model.load_state_dict({k: v for k, v in initial_weights.items()
                                           if k in model.state_dict() and v.shape == model.state_dict()[k].shape}, strict=False)
            if loaded.unexpected_keys or any(not k.startswith(('candidates.', 'routing.')) for k in loaded.missing_keys):
                raise ValueError('Causal initialization missed required context weights')
            if args.robust_semantics:
                dimension = model.dimension
                old_modes = initial_weights['routing.bias'].numel()
                if args.modes % old_modes:
                    raise ValueError('Expanded candidates must replicate old continuous modes')
                with torch.no_grad():
                    for i in range(args.modes // old_modes):
                        sl = slice(i * old_modes, (i + 1) * old_modes)
                        model.candidates[-1].weight.reshape(args.modes, dimension, -1)[sl].copy_(
                            initial_weights['candidates.2.weight'].reshape(old_modes, dimension, -1)
                            + (torch.zeros_like(initial_weights['candidates.2.weight']) if args.opening_semantics
                               else 0.001 * torch.randn_like(initial_weights['candidates.2.weight'])).reshape(old_modes, dimension, -1))
                        model.candidates[-1].bias.reshape(args.modes, dimension)[sl].copy_(
                            initial_weights['candidates.2.bias'].reshape(old_modes, dimension))
                        model.routing.weight[sl].copy_(initial_weights['routing.weight'])
                        model.routing.bias[sl].copy_(initial_weights['routing.bias'] - math.log(args.modes / old_modes))
            model = model.to(device)
        else:
            buffers, subspace = token_subspace(prefix[training_mask], lengths[training_mask])
            model = AutoregressivePrior(*buffers).to(device)
        info.update(subspace=subspace, trainable_parameters=sum(p.numel() for p in model.parameters()))
        if info['trainable_parameters'] > 25_000_000:
            raise ValueError('Reader exceeds compact model budget')
        print(json.dumps({'subspace': subspace, 'parameters': info['trainable_parameters']}), flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0)
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
    prototype = None
    opening_prototype = None
    if args.robust_semantics:
        names = [r['category'] for r in corpus['rows']]
        categories = sorted(set(names))
        category_ids = torch.tensor([categories.index(k) for k in names], device=device)
        means = torch.stack([features[training_mask & (category_ids == i)].mean(0) for i in range(len(categories))])
        prototype = torch.nn.functional.normalize(means, dim=-1)[category_ids]
        if args.opening_semantics:
            names = [f"{r['category']}:{r['opening_index']}" for r in corpus['rows']]
            openings = sorted(set(names))
            opening_ids = torch.tensor([openings.index(k) for k in names], device=device)
            means = torch.stack([features[training_mask & (opening_ids == i)].mean(0) for i in range(len(openings))])
            opening_prototype = torch.nn.functional.normalize(means, dim=-1)[opening_ids]
    args.run_dir.mkdir(parents=True, exist_ok=True)
    with (args.run_dir / 'loss.jsonl').open('a') as log:
        for step in range(start, args.steps):
            chosen = training[torch.randint(len(training), (16,), device=device)]
            lr = learning_rate * min(1, (step + 1) / 100) * (0.1 + 0.45 * (1 + math.cos(math.pi * step / schedule_steps)))
            for group in optimizer.param_groups:
                group['lr'] = lr
            optimizer.zero_grad(set_to_none=True)
            objective = multihypothesis_loss if multi else causal_loss
            semantic = features[chosen]
            if args.robust_semantics:
                semantic = semantic.clone()
                kinds = torch.rand(len(chosen), device=device)
                if args.opening_semantics:
                    opening_mask = kinds < 0.45
                    semantic[opening_mask] = torch.nn.functional.normalize(opening_prototype[chosen][opening_mask]
                        + 0.01 * torch.randn_like(semantic[opening_mask]), dim=-1)
                    mean_mask = (kinds >= 0.45) & (kinds < 0.7)
                    jitter_mask, zero_mask = torch.zeros_like(mean_mask), kinds >= 0.95
                else:
                    mean_mask, jitter_mask, zero_mask = kinds < 0.5, (kinds >= 0.8) & (kinds < 0.9), kinds >= 0.9
                semantic[mean_mask] = torch.nn.functional.normalize(prototype[chosen][mean_mask]
                    + 0.01 * torch.randn_like(semantic[mean_mask]), dim=-1)
                semantic[jitter_mask] = torch.nn.functional.normalize(semantic[jitter_mask]
                    + 0.015 * torch.randn_like(semantic[jitter_mask]), dim=-1)
                semantic[zero_mask] = 0
            loss, components = objective(model, semantic, prefix[chosen], lengths[chosen])
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
