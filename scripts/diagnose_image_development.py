"""Frozen 2x2 reader/length development ablation; no fresh-test acceptance."""
import argparse
import json
from pathlib import Path

import torch
from image_probe import make_reader
from safetensors.torch import load_file

from latent_diaries.image_query_reader import ridge_length
from latent_diaries.inversion import digest, load_backbone, render


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--design', type=Path, required=True)
    parser.add_argument('--baseline-dir', type=Path, required=True)
    parser.add_argument('--query-dir', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if not Path('/.dockerenv').exists() or args.output_dir.exists():
        raise ValueError('Use managed worker and a fresh output directory')
    rows = json.loads(args.dataset.read_text())['rows']
    design = {r['id']: r for r in json.loads(args.design.read_text())['rows']}
    features = torch.tensor([r['features'] for r in rows], dtype=torch.float32)
    train = torch.tensor([r['split'] == 'train' for r in rows])
    train_rows = [r for r in rows if r['split'] == 'train']
    training_lengths = torch.tensor([
        load_file(str(Path(r['cache_dir']) / 'prefix.safetensors'))['soft_tokens'].shape[1]
        for r in train_rows], dtype=torch.float32)
    ridge, ridge_info = ridge_length(features[train], training_lengths)
    configs, models, fingerprints = {}, {}, {}
    for name, folder in [('global', args.baseline_dir), ('query', args.query_dir)]:
        configs[name] = json.loads((folder / 'config.json').read_text())
        path = folder / 'reader.safetensors'
        fingerprints[name] = digest(path)
        assert fingerprints[name] == configs[name]['reader_sha256']
        models[name] = make_reader(configs[name], load_file(str(path))).eval().requires_grad_(False)
    coefs = {'ols': models['global'].length_coef.clone(), 'ridge': ridge}
    wrapper, builder, speaker = load_backbone(configs['query'])
    targets = []
    with torch.inference_mode():
        for row in rows:
            ids = wrapper._tokenize_texts([wrapper._build_assistant_text(design[row['id']]['teacher_text'])])[0]
            targets.append(wrapper.model.talker.text_projection(
                wrapper.model.talker.get_text_embeddings()(ids[:, 3:-5]))[0].cpu())
    # The shared final Linear(width,1024) limits centered output rank to width.
    # Compute a training-only best-rank lower bound before interpreting fit loss.
    with torch.inference_mode():
        training_matrix = torch.cat([t for i, t in enumerate(targets) if bool(train[i])]).double().to(speaker.device)
        query_model = models['query']
        normalized = (training_matrix - query_model.mean.to(speaker.device)) / query_model.scale.to(speaker.device)
        normalized = normalized - normalized.mean(0)
        singular = torch.linalg.svdvals(normalized)
        width = configs['query']['width']
        rank_floor = float(singular[width:].square().sum() / normalized.numel())
    results, summary, predictions = [], {}, {}
    with torch.inference_mode():
        for name, model in models.items():
            prefix = model(features)[0]
            predictions[name] = prefix
            before = prefix.clone()
            model.length_coef.copy_(ridge)
            assert torch.equal(before, model(features)[0])
            for method, coef in coefs.items():
                length = (torch.cat([torch.ones(len(rows), 1), features], -1) @ coef).round().long()
                length = length.clamp(1, model.maximum_tokens)
                selected = []
                for i, row in enumerate(rows):
                    n = min(targets[i].shape[0], prefix.shape[1])
                    aligned = min(n, int(length[i]))
                    item = {'id': row['id'], 'split': row['split'],
                            'valid_prefix_mse': float((prefix[i, :n]-targets[i][:n]).square().mean()),
                            'aligned_prefix_mse': float((prefix[i, :aligned]-targets[i][:aligned]).square().mean()),
                            'length_matches': int(length[i]) == targets[i].shape[0],
                            'absolute_length_error': abs(int(length[i]) - targets[i].shape[0])}
                    selected.append(item)
                key = name+'_'+method
                results.append({'condition': key, 'rows': selected})
                summary[key] = {}
                for split in ['train', 'heldout']:
                    subset = [r for r in selected if r['split'] == split]
                    summary[key][split] = {k: sum(r[k] for r in subset)/len(subset)
                                          for k in ['valid_prefix_mse', 'aligned_prefix_mse',
                                                    'length_matches', 'absolute_length_error']}
    threshold = 1e-5
    passed = summary['query_ridge']['heldout']['valid_prefix_mse'] < threshold
    args.output_dir.mkdir(parents=True)
    generated = []
    if passed:
        plan = json.loads(args.plan.read_text())
        chosen = plan['heldout_ids'] + [plan['training_id']]
        model = models['query'].to(speaker.device)
        for label in chosen:
            i = next(i for i, r in enumerate(rows) if r['id'] == label)
            with torch.inference_mode():
                p, _, length = model(features[i:i+1].to(speaker.device))
            generated.append({'id': label, **render(builder, wrapper, p[:, :int(length[0])], speaker,
                                                     args.output_dir / f'{label}.wav', 105, 42)})
    for name, folder in [('global', args.baseline_dir), ('query', args.query_dir)]:
        assert digest(folder / 'reader.safetensors') == fingerprints[name]
    data = {'scope': 'Exposed-development only', 'weights_updated': False,
            'in_memory_length_head_overrides': True,
            'shared_output_rank_training_mse_floor': rank_floor,
            'ridge_training_only': ridge_info, 'summary': summary, 'results': results,
            'development_audio_gate_passed': passed, 'gate_threshold': threshold,
            'human_quality_acceptance': False, 'generated': generated,
            'source_sha256': digest(Path(__file__)), 'reader_fingerprints': fingerprints}
    (args.output_dir / 'diagnostic.json').write_text(json.dumps(data, indent=2)+'\n')
    print(json.dumps({'summary': summary, 'audio_gate_passed': passed, 'rank_floor': rank_floor}), flush=True)


if __name__ == '__main__':
    main()
