"""Frozen teacher-forced conditioning-distribution diagnostic; not human quality."""
import argparse
import json
from pathlib import Path

import torch
from evaluate_autoregressive_prior import load_prior
from safetensors.torch import load_file

from latent_diaries.inversion import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--corpus-dir', type=Path, required=True)
    parser.add_argument('--features-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not Path('/.dockerenv').exists() or args.output.exists():
        raise ValueError('Use managed worker and fresh diagnostic output')
    info = json.loads((args.run_dir / 'config.json').read_text())
    file = args.run_dir / 'reader.safetensors';assert digest(file) == info['reader_sha256']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_prior(load_file(str(file)), 'multihypothesis', info.get('modes', 8)).to(device).eval().requires_grad_(False)
    corpus = json.loads((args.corpus_dir / 'corpus.json').read_text())['rows']
    prefix = load_file(str(args.corpus_dir / 'prefixes.safetensors'))['prefixes'].to(device)
    features = load_file(str(args.features_dir / 'features.safetensors'))['embeddings'].to(device)
    lengths = torch.tensor([r['length'] for r in corpus], device=device)
    categories = sorted({r['category'] for r in corpus})
    means = {category: torch.nn.functional.normalize(features[torch.tensor([
        r['split'] == 'train' and r['category'] == category for r in corpus], device=device)].mean(0), dim=0)
        for category in categories}
    selected = [i for split in ['train', 'heldout'] for category in categories
                for i in [i for i, row in enumerate(corpus) if row['split'] == split and row['category'] == category][:4]]
    records = []
    with torch.inference_mode():
        for index in selected:
            clean = model.encode(prefix[index:index + 1])
            previous = torch.cat([torch.zeros_like(clean[:, :1]), clean], 1)
            length = int(lengths[index])
            for mode in ['exact', 'prototype', 'zero']:
                semantic = features[index:index + 1] if mode == 'exact' else (
                    means[corpus[index]['category']][None] if mode == 'prototype' else torch.zeros_like(features[:1]))
                contexts = model.contexts(semantic, previous)
                candidates, logits = model.predictions(contexts[:, :length])
                routed = candidates[0, torch.arange(length, device=device), logits[0].argmax(-1)]
                decoded = model.decode(routed)
                raw = float((decoded - prefix[index, :length]).square().mean())
                stop = model.stop(contexts[0, :length + 1]).sigmoid().squeeze(-1)
                records.append({'id': corpus[index]['id'], 'split': corpus[index]['split'], 'mode': mode,
                                'routed_gold_history_raw_prefix_mse': raw,
                                'stop_at_gold_eos': float(stop[length]),
                                'premature_stop_positions': int((stop[:length] >= 0.5).sum())})
    summary = {}
    for split in ['train', 'heldout']:
        summary[split] = {}
        for mode in ['exact', 'prototype', 'zero']:
            rows = [r for r in records if r['split'] == split and r['mode'] == mode]
            summary[split][mode] = {'count': len(rows), **{key: sum(r[key] for r in rows) / len(rows)
                for key in ['routed_gold_history_raw_prefix_mse', 'stop_at_gold_eos', 'premature_stop_positions']}}
    result = {'scope': 'Fixed40 train/40 held text-combination diagnostics; gold history, not autoregressive quality',
              'summary': summary, 'records': records, 'reader_sha256': info['reader_sha256'],
              'source_sha256': digest(Path(__file__)), 'weights_updated': False}
    assert digest(file) == info['reader_sha256']
    args.output.write_text(json.dumps(result, indent=2) + '\n');args.output.chmod(0o600)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
