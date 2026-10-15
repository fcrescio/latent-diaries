"""Select affine semantic gain using training-group continuous opening likelihood."""
import argparse
import json
import random
from pathlib import Path

import torch
from evaluate_autoregressive_prior import load_prior
from safetensors.torch import load_file, save_file
from torch.nn.functional import normalize

from latent_diaries.inversion import digest
from latent_diaries.prior_calibration import opening_nll, shared_opening_length
from latent_diaries.semantic_adapter import fit_coefficient, scale_semantic_deviation
from latent_diaries.visual_anchors import training_anchor_targets


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['image-dataset', 'labels', 'corpus-dir', 'features', 'prior-dir', 'initial-adapter', 'output-dir']:
        p.add_argument('--' + name, type=Path, required=True)
    args = p.parse_args()
    if not Path('/.dockerenv').exists() or args.output_dir.exists():
        raise ValueError('Use managed worker and fresh output')
    protocol = json.loads(Path('configs/research/semantic-gain-v15.json').read_text())
    initial = json.loads((args.initial_adapter.parent / 'manifest.json').read_text())
    prior_info = json.loads((args.prior_dir / 'config.json').read_text())
    prior_file = args.prior_dir / 'reader.safetensors'
    corpus_file = args.corpus_dir / 'corpus.json'
    prefix_file = args.corpus_dir / 'prefixes.safetensors'
    corpus = json.loads(corpus_file.read_text())
    if (digest(args.image_dataset) != initial['image_dataset_sha256'] or digest(args.labels) != initial['anchor_labels_sha256']
            or digest(corpus_file) != initial['corpus_sha256'] or digest(args.features) != initial['features_sha256']
            or digest(args.initial_adapter) != initial['adapter_sha256'] or digest(prior_file) != prior_info['reader_sha256']
            or digest(prefix_file) != corpus['provenance']['prefix_sha256']
            or digest(Path('uv.lock')) != prior_info['lock_sha256']
            or any(initial[k] != prior_info[k] for k in ['corpus_sha256', 'features_sha256'])):
        raise ValueError('Frozen caches/weights differ')
    rows = [r for r in json.loads(args.image_dataset.read_text())['rows'] if r['split'] == 'train']
    labels = training_anchor_targets([json.loads(s) for s in args.labels.read_text().splitlines() if s], {r['id'] for r in rows})
    rows = [r for r in rows if labels.get(r['id']) is not None]
    keys = sorted({labels[r['id']] for r in rows})
    by_text = {k: [i for i, r in enumerate(corpus['rows']) if r['split'] == 'train'
                  and f"{r['category']}:{r['opening_index']}" == k] for k in keys}
    embeddings = load_file(str(args.features))['embeddings']
    means = {k: normalize(embeddings[by_text[k]].mean(0), dim=0) for k in keys}
    x = torch.tensor([r['features'] for r in rows])
    y = torch.stack([means[labels[r['id']]] for r in rows])
    groups = sorted({r['scene_group'] for r in rows})
    if len(groups) < 5:
        raise ValueError('Insufficient independent training groups')
    random.Random(42).shuffle(groups)
    folds = {g: i % 5 for i, g in enumerate(groups)}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    prior = load_prior(load_file(str(prior_file)), prior_info['recipe'], prior_info['modes']).to(device).eval().requires_grad_(False)
    prefixes = load_file(str(prefix_file))['prefixes'].to(device)
    lengths = torch.tensor([r['length'] for r in corpus['rows']], device=device)
    boundaries = {k: shared_opening_length(prefixes[ids], lengths[ids]) for k, ids in by_text.items()}
    maximum = max(boundaries.values())
    gold = prefixes[[by_text[labels[r['id']]][0] for r in rows], :maximum]
    opening_lengths = torch.tensor([boundaries[labels[r['id']]] for r in rows], device=device)
    predictions = {gain: torch.zeros(len(rows)) for gain in protocol['gain_grid']}
    with torch.inference_mode():
        for fold in range(5):
            held = torch.tensor([folds[r['scene_group']] == fold for r in rows])
            coefficient = fit_coefficient(x[~held], y[~held], initial['alpha'])
            mean = y[~held].double().mean(0).float()
            query = torch.cat([x[held].new_ones(int(held.sum()), 1), x[held]], -1).to(device)
            mask = held.to(device)
            for gain in protocol['gain_grid']:
                adapted = scale_semantic_deviation(coefficient, mean, gain).to(device)
                value = opening_nll(prior, normalize(query @ adapted, dim=-1), gold[mask], opening_lengths[mask])
                predictions[gain][held] = value.cpu()
            print(json.dumps({'fold_completed': fold + 1, 'validation_images': int(held.sum())}), flush=True)
    macro, micro = {}, {}
    for gain, value in predictions.items():
        macro[str(gain)] = sum(float(value[[labels[r['id']] == k for r in rows]].mean()) for k in keys) / len(keys)
        micro[str(gain)] = float(value.mean())
    selected = min(protocol['gain_grid'], key=lambda g: (macro[str(g)], g))
    base = load_file(str(args.initial_adapter))['coefficient']
    coefficient = scale_semantic_deviation(base, y.double().mean(0).float(), selected)
    args.output_dir.mkdir(parents=True)
    file = args.output_dir / 'adapter.safetensors'
    save_file({'coefficient': coefficient.contiguous()}, str(file))
    info = initial | {'scope': 'Training-group likelihood-selected semantic deviation gain', 'gain': selected,
                      'gain_macro_opening_nll': macro, 'gain_micro_opening_nll': micro,
                      'initial_adapter_sha256': digest(args.initial_adapter), 'prior_sha256': digest(prior_file),
                      'prior_config_sha256': digest(args.prior_dir / 'config.json'),
                      'source_sha256': digest(Path(__file__)), 'calibration_source_sha256': digest(Path('src/latent_diaries/semantic_adapter.py')),
                      'loss_source_sha256': digest(Path('src/latent_diaries/prior_calibration.py')),
                      'protocol_sha256': digest(Path('configs/research/semantic-gain-v15.json')),
                      'adapter_sha256': digest(file), 'peak_allocated_vram_bytes': torch.cuda.max_memory_allocated() if device.type == 'cuda' else None}
    (args.output_dir / 'manifest.json').write_text(json.dumps(info, indent=2) + '\n')
    print(json.dumps({'gain': selected, 'macro_nll': macro, 'micro_nll': micro}), flush=True)


if __name__ == '__main__':
    main()
