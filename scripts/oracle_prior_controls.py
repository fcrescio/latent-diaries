"""Four offline oracle controls: gold-history routing versus gold best candidate."""
import argparse
import json
import os
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from latent_diaries.inversion import digest, load_backbone, render
from latent_diaries.multihypothesis_prior import MultihypothesisPrior
from latent_diaries.prior_diagnostics import trace_prior


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ladder-results', type=Path, required=True)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--corpus-dir', type=Path, required=True)
    p.add_argument('--features-dir', type=Path, required=True)
    p.add_argument('--ids', nargs='+', required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    if not Path('/.dockerenv').exists() or args.output_dir.exists():
        raise ValueError('Managed worker and fresh oracle directory required')
    ladder = json.loads(args.ladder_results.read_text())
    info = json.loads((args.run_dir / 'config.json').read_text())
    files = {'reader': args.run_dir / 'reader.safetensors', 'corpus': args.corpus_dir / 'corpus.json',
             'prefix': args.corpus_dir / 'prefixes.safetensors', 'features': args.features_dir / 'features.safetensors'}
    for key, file in files.items():
        if digest(file) != ladder['hashes'][key] or digest(file) != info[key + '_sha256']:
            raise ValueError('Historical training artifact differs')
    permitted = {r['image_id'] for r in ladder['results'] if r.get('level') == 'L1'}
    if not set(args.ids) <= permitted or info['lock_sha256'] != digest(Path('uv.lock')):
        raise ValueError('Existing training-oracle scope only')
    corpus = json.loads(files['corpus'].read_text())['rows']
    features = load_file(str(files['features']))['embeddings']
    targets = load_file(str(files['prefix']))['prefixes']
    wrapper, builder, speaker = load_backbone(info)
    weights = load_file(str(files['reader']))
    model = MultihypothesisPrior(*(weights[k] for k in ['mean', 'scale', 'basis', 'latent_scale']), modes=info['modes'])
    model.load_state_dict(weights);model.to(speaker.device).eval().requires_grad_(False)
    args.output_dir.mkdir(mode=0o700)
    results = []
    with torch.inference_mode():
        for image_id in args.ids:
            index = next(i for i, r in enumerate(corpus) if r['category'] == 'image_' + image_id and r['split'] == 'train')
            gold = targets[index:index+1, :corpus[index]['length']].to(speaker.device)
            semantic = features[index:index+1].to(speaker.device)
            latent = model.encode(gold)
            contexts = model.contexts(semantic, torch.cat([torch.zeros_like(latent[:, :1]), latent], 1))[:, :-1]
            candidates, _ = model.predictions(contexts)
            error = ((candidates - latent[:, :, None]) * model.latent_scale).square().sum(-1) / model.mean.numel()
            indices = error.argmin(-1)
            best = candidates.gather(2, indices[:, :, None, None].expand(-1, -1, 1, model.dimension)).squeeze(2)
            controls = {'gold_history_routed': trace_prior(model, semantic, gold, 42, gold_history=True)['prefix'],
                        'gold_history_best_candidate': model.decode(best)}
            for name, prefix in controls.items():
                file = args.output_dir / f'{image_id}-{name}.safetensors'
                save_file({'prefix': prefix.cpu().contiguous()}, str(file))
                result = {'image_id': image_id, 'level': name, 'seed': 42, 'oracle_text_used_offline': True,
                    'deployable_image_inference': False, 'prefix_sha256': digest(file),
                    'mean_normalized_prefix_mse': float(((prefix-gold)/model.scale).square().mean()),
                    'human_quality': 'unrated', **render(builder, wrapper, prefix, speaker,
                        args.output_dir / f'{image_id}-{name}.wav', ladder['max_frames'], 42)}
                results.append(result);print(json.dumps(result), flush=True)
    manifest = {'scope': 'Offline oracle gold histories and oracle candidate choice; not deployable or a new candidate',
        'ladder_results_sha256': digest(args.ladder_results), 'source_revision': os.environ.get('LD_CODE_REVISION'),
        'source_sha256': digest(Path(__file__)), 'historical_hashes': {k: digest(f) for k, f in files.items()},
        'model_revision': info['model_revision'], 'lock_sha256': digest(Path('uv.lock')), 'results': results}
    (args.output_dir / 'results.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
