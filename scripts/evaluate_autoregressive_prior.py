"""Numeric-only causal prefix generation; distinguish text-feature diagnostics."""
import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from latent_diaries.autoregressive_prior import AutoregressivePrior
from latent_diaries.image_reader import validate_image_inputs
from latent_diaries.inversion import digest, load_backbone, render
from latent_diaries.multihypothesis_prior import MultihypothesisPrior


def load_prior(weights, recipe='causal-flow', modes=8):
    reader = MultihypothesisPrior if recipe == 'multihypothesis' else AutoregressivePrior
    kwargs = {'modes': modes} if recipe == 'multihypothesis' else {}
    model = reader(*(weights[k] for k in ['mean', 'scale', 'basis', 'latent_scale']), **kwargs)
    model.load_state_dict(weights)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--ids', nargs='+', required=True)
    parser.add_argument('--mode', choices=['image', 'training-text-control'], required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--adapter', type=Path)
    parser.add_argument('--greedy-candidates', action='store_true')
    parser.add_argument('--guidance-scale', type=float, default=0.0)
    parser.add_argument('--guidance-steps', type=int, default=12)
    args = parser.parse_args()
    if not Path('/.dockerenv').exists() or args.output_dir.exists():
        raise ValueError('Use managed worker and new output directory')
    info = json.loads((args.run_dir / 'config.json').read_text())
    if not 0 <= args.guidance_scale <= 2 or not 0 <= args.guidance_steps <= 32:
        raise ValueError('Guidance outside bounded research protocol')
    if args.guidance_scale and not info.get('robust_semantics'):
        raise ValueError('Guidance requires a prior trained on missing semantic conditions')
    file = args.run_dir / 'reader.safetensors'
    if digest(file) != info['reader_sha256'] or digest(Path('uv.lock')) != info['lock_sha256']:
        raise ValueError('Frozen reader or lock differs')
    rows = json.loads(args.inputs.read_text())['rows']
    validate_image_inputs(rows, [r['id'] for r in rows], allow_new=True)
    if not set(args.ids) <= {r['id'] for r in rows} or len(set(args.ids)) != len(args.ids):
        raise ValueError('Missing/duplicate requested IDs')
    model = load_prior(load_file(str(file)), info.get('recipe', 'causal-flow'), info.get('modes', 8))
    if args.adapter:
        adapter_info = json.loads((args.adapter.parent / 'manifest.json').read_text())
        if (digest(args.adapter) != adapter_info['adapter_sha256'] or
                adapter_info['corpus_sha256'] != info['corpus_sha256'] or
                adapter_info['features_sha256'] != info['features_sha256'] or
                (adapter_info.get('prior_sha256') and adapter_info['prior_sha256'] != info['reader_sha256'])):
            raise ValueError('Image adapter and language-prior training provenance differ')
    wrapper, builder, speaker = load_backbone(info)
    model.to(speaker.device).eval().requires_grad_(False)
    adapter = load_file(str(args.adapter))['coefficient'].to(speaker.device) if args.adapter else None
    chosen = [r for r in rows if r['id'] in args.ids]
    chosen.append({'id': 'null', 'split': 'control', 'features': [0.0] * 768})
    args.output_dir.mkdir(parents=True)
    results = []
    for row in chosen:
        features = torch.tensor([row['features']], device=speaker.device, dtype=torch.float32)
        if adapter is not None:
            features = torch.nn.functional.normalize(torch.cat([features.new_ones(1, 1), features], -1) @ adapter, dim=-1)
        kwargs = {'sample_modes': not args.greedy_candidates} if info.get('robust_semantics') else {}
        if args.guidance_scale:
            kwargs.update(guidance_scale=args.guidance_scale, guidance_steps=args.guidance_steps)
        prefix, stop_probabilities = model.sample(features, seed=42, **kwargs)
        result = {'id': row['id'], 'split': row['split'], 'predicted_prefix_length': prefix.shape[1],
                  'prefix_stop_probabilities': stop_probabilities, 'human_listening_status': 'pending',
                  **render(builder, wrapper, prefix, speaker, args.output_dir / f"{row['id']}.wav", 105, 42)}
        results.append(result);print(json.dumps(result), flush=True)
    assert digest(file) == info['reader_sha256']
    text_control = args.mode != 'image'
    manifest = {'scope': 'Text-feature diagnostic' if text_control else 'Exposed development only',
                'text_used': text_control, 'image_conditioning': not text_control,
                'teacher_files_read': False, 'linguistic_labels_read': False,
                'reader_sha256': info['reader_sha256'], 'inputs_sha256': digest(args.inputs),
                'adapter_sha256': digest(args.adapter) if args.adapter else None,
                'continuous_modes_sampled': bool(info.get('robust_semantics') and not args.greedy_candidates),
                'guidance_scale': args.guidance_scale, 'guidance_steps': args.guidance_steps,
                'prior_source_sha256': digest(Path('src/latent_diaries/multihypothesis_prior.py')),
                'source_sha256': digest(Path(__file__)), 'results': results}
    (args.output_dir / 'results.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
