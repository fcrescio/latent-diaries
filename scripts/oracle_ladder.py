"""No-training offline L0-L5 localization; oracle text never labels image inference."""
import argparse
import json
import os
from pathlib import Path

import soundfile as sf
import torch
from safetensors.torch import load_file, save_file

from latent_diaries.feature_provenance import verify_image_features
from latent_diaries.inversion import digest, load_backbone, render
from latent_diaries.multihypothesis_prior import MultihypothesisPrior
from latent_diaries.prior_diagnostics import compare_traces, trace_prior
from latent_diaries.semantic_adapter import select_alpha, semantic_condition


def scalar_geometry(query, candidates, own_indices):
    similarity = query @ candidates.T
    own = similarity[own_indices].max()
    other = similarity[~own_indices].max()
    return {'own_max_cosine': float(own), 'other_max_cosine': float(other),
            'own_retrieved': bool(own > other), 'neighbor_margin': float(own - other),
            'nearest_normalized_distance': float((2 - 2 * similarity.max()).clamp_min(0).sqrt())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--corpus-dir', type=Path, required=True)
    parser.add_argument('--features-dir', type=Path, required=True)
    parser.add_argument('--adapter-dir', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--image-cache', type=Path, action='append', required=True)
    parser.add_argument('--ids', nargs='+', required=True)
    parser.add_argument('--development-ids', nargs='*', default=[])
    parser.add_argument('--seeds', nargs='+', type=int, default=[41, 42, 43])
    parser.add_argument('--max-frames', type=int, default=160)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if not Path('/.dockerenv').exists() or args.output_dir.exists():
        raise ValueError('Managed worker and new private output directory required')
    info = json.loads((args.run_dir / 'config.json').read_text())
    corpus_file = args.corpus_dir / 'corpus.json'
    corpus = json.loads(corpus_file.read_text())
    features_file = args.features_dir / 'features.safetensors'
    feature_info = json.loads((args.features_dir / 'manifest.json').read_text())
    adapter_file = args.adapter_dir / 'adapter.safetensors'
    adapter_info = json.loads((args.adapter_dir / 'manifest.json').read_text())
    paths = {'reader': args.run_dir / 'reader.safetensors', 'corpus': corpus_file,
             'features': features_file, 'prefix': args.corpus_dir / 'prefixes.safetensors',
             'adapter': adapter_file, 'dataset': args.dataset}
    hashes = {key: digest(path) for key, path in paths.items()}
    for key in ['reader', 'corpus', 'features', 'prefix']:
        if hashes[key] != info[key + '_sha256']:
            raise ValueError('Prior training artifact changed: ' + key)
    for key in ['corpus', 'features', 'adapter']:
        if hashes[key] != adapter_info[key + '_sha256']:
            raise ValueError('Adapter provenance differs: ' + key)
    if (feature_info['ids'] != [r['id'] for r in corpus['rows']]
            or feature_info['corpus_sha256'] != hashes['corpus']
            or feature_info['features_sha256'] != hashes['features']
            or digest(Path('uv.lock')) != info['lock_sha256']):
        raise ValueError('Cache order or locked runtime differs')
    image_rows = json.loads(args.dataset.read_text())['rows']
    if any(r['split'] not in {'train', 'heldout'} for r in image_rows):
        raise ValueError('Fresh input forbidden')
    by_image = {r['id']: r for r in image_rows}
    if len(set(args.ids)) != len(args.ids) or len(args.ids) < 2:
        raise ValueError('At least two unique training images for shuffled controls')
    if any(by_image[k]['split'] != 'train' for k in args.ids):
        raise ValueError('Oracle labels restricted to training images')
    if any(by_image[k]['split'] != 'heldout' for k in args.development_ids):
        raise ValueError('Development controls must be image-only heldout')
    evidence = verify_image_features(image_rows, args.image_cache)
    args.output_dir.mkdir(mode=0o700)
    vectors = load_file(str(features_file))['embeddings']
    prefixes = load_file(str(paths['prefix']))['prefixes']
    training_mask = torch.tensor([r['split'] == 'train' for r in corpus['rows']])
    names = [r['category'] for r in corpus['rows']]
    categories = sorted(set(names))
    means_raw = torch.stack([vectors[training_mask & torch.tensor([n == k for n in names])].mean(0)
                             for k in categories])
    means = torch.nn.functional.normalize(means_raw, dim=-1)
    centers = {name: means[i] for i, name in enumerate(categories)}
    mean = vectors[training_mask].mean(0)
    mean_file = args.output_dir / 'training-mean.safetensors'
    save_file({'mean': mean}, str(mean_file))
    mean_file.with_suffix('.json').write_text(json.dumps({'corpus_sha256': hashes['corpus'],
        'features_sha256': hashes['features'], 'mean_sha256': digest(mean_file), 'training_rows_only': True}) + '\n')
    coefficient = load_file(str(adapter_file))['coefficient']
    fit_rows = [r for r in image_rows if r['split'] == 'train' and 'image_' + r['id'] in centers]
    x = torch.tensor([r['features'] for r in fit_rows])
    target = torch.stack([centers['image_' + r['id']] for r in fit_rows])
    alpha, cv = select_alpha(x, target, [r['scene_group'] for r in fit_rows],
                            alphas=(0.001, 0.01, 0.1, 1, 10, 100), folds=5, seed=42)
    # Offline characterization only: never refit or replace the historical adapter.
    geometry = []
    d1 = vectors[training_mask]
    train_names = [n for n, keep in zip(names, training_mask, strict=True) if keep]
    for row in fit_rows:
        group = 'image_' + row['id']
        oracle = centers[group]
        raw = torch.cat([torch.ones(1), torch.tensor(row['features'])]) @ coefficient
        predicted = torch.nn.functional.normalize(raw, dim=-1)
        group_mask = torch.tensor([n == group for n in train_names])
        group_index = categories.index(group)
        geometry.append({'id': row['id'], 'd2_d3_cosine': float(oracle @ predicted),
            'd2_d3_distance': float((oracle - predicted).norm()), 'd3_pre_normalization_norm': float(raw.norm()),
            'd2_pre_normalization_norm': float(means_raw[group_index].norm()),
            'd1_d2_mean_cosine': float((d1[group_mask] @ oracle).mean()),
            'd3_to_d1': scalar_geometry(predicted, d1, group_mask),
            'd3_to_d2': scalar_geometry(predicted, means, torch.tensor([n == group for n in categories]))})
    (args.output_dir / 'representation.json').write_text(json.dumps({'training_images': len(fit_rows),
        'extended_alpha_selected': alpha, 'extended_cv': cv, 'historical_adapter_unchanged': True,
        'training_mean_norm': float(mean.norm()), 'd2_between_group_variance': float(means.var(0).sum()),
        'rows': geometry}, indent=2) + '\n')
    wrapper, builder, speaker = load_backbone(info)
    weights = load_file(str(paths['reader']))
    model = MultihypothesisPrior(*(weights[k] for k in ['mean', 'scale', 'basis', 'latent_scale']), modes=info['modes'])
    model.load_state_dict(weights);model.to(speaker.device).eval().requires_grad_(False)
    coefficient = coefficient.to(speaker.device)
    voice_prompt = wrapper.create_voice_clone_prompt(ref_audio=os.environ['LD_REF_AUDIO'], x_vector_only_mode=True)
    torch.testing.assert_close(voice_prompt[0].ref_spk_embedding.to(speaker.device), speaker[0], rtol=0, atol=0)
    results, traces = [], []
    decoding = {'max_new_tokens': args.max_frames + 1, 'min_new_tokens': 2, 'do_sample': True,
                'top_k': 50, 'top_p': 1.0, 'temperature': 0.9, 'repetition_penalty': 1.05,
                'subtalker_dosample': True, 'subtalker_top_k': 50, 'subtalker_top_p': 1.0, 'subtalker_temperature': 0.9}
    with torch.inference_mode():
        for image_id in args.ids:
            group = 'image_' + image_id
            indices = [i for i, r in enumerate(corpus['rows']) if r['category'] == group and r['split'] == 'train']
            if not indices:
                raise ValueError('Training image lacks existing narrative supervision')
            index = indices[0];row = corpus['rows'][index]
            exact = vectors[index:index+1].to(speaker.device)
            oracle = centers[group][None].to(speaker.device)
            image = torch.tensor([by_image[image_id]['features']], device=speaker.device)
            predicted = semantic_condition(image, coefficient)
            gold = prefixes[index:index+1, :row['length']].to(speaker.device)
            for seed in args.seeds:
                for level, semantic in [('L2', exact), ('L3', oracle), ('L4', predicted)]:
                    gold_trace = trace_prior(model, semantic, gold, seed, gold_history=True)
                    free = trace_prior(model, semantic, gold, seed)
                    interventions = {}
                    for position in range(gold.shape[1]):
                        reset = trace_prior(model, semantic, gold, seed, reset_position=position)
                        interventions[str(position)] = compare_traces(gold_trace, reset)
                    traces.append({'image_id': image_id, 'seed': seed, 'level': level,
                        'gold_history': compare_traces(gold_trace, gold_trace),
                        'free_running': compare_traces(gold_trace, free), 'single_oracle_vector_resets': interventions,
                        'scope': 'Aligned diagnostic ignores stopping; actual audio uses unmodified sampler'})
                for level, semantic in [('L0', None), ('L1', None), ('L2', exact), ('L3', oracle), ('L4', predicted)]:
                    path = args.output_dir / f'{image_id}-{level}-s{seed}.wav'
                    if level == 'L0':
                        captured = {};original = wrapper.model.talker.generate
                        def stock_generate(_captured=captured, _original=original, **kwargs):
                            _captured['prefill'] = {k: kwargs[k].detach().clone() for k in
                                ['inputs_embeds', 'attention_mask', 'trailing_text_hidden', 'tts_pad_embed']}
                            output = _original(**kwargs)
                            _captured['frames'] = sum(h[-1] is not None for h in output.hidden_states)
                            return output
                        wrapper.model.talker.generate = stock_generate
                        try:
                            with torch.random.fork_rng(devices=[speaker.device.index or 0]):
                                torch.manual_seed(seed)
                                wavs, sr = wrapper.generate_voice_clone(text=row['text'], language='Italian',
                                    voice_clone_prompt=voice_prompt, non_streaming_mode=True, **decoding)
                        finally:
                            wrapper.model.talker.generate = original
                        from latent_diaries.inversion import condition
                        assembled = builder.build_prefill(condition(gold, speaker))
                        parity = {k: float((assembled[k].float() - value.float()).abs().max())
                                  for k, value in captured['prefill'].items()}
                        sf.write(path, wavs[0], sr)
                        result = {'frames_including_stock_stop': captured['frames'], 'sample_rate': sr,
                                  'seconds': len(wavs[0])/sr, 'audio': str(path), 'audio_sha256': digest(path),
                                  'stock_continuous_prefill_max_errors': parity}
                    else:
                        prefix, stop = (gold, None) if level == 'L1' else model.sample(semantic, seed=seed, sample_modes=True)
                        result = {**render(builder, wrapper, prefix, speaker, path, args.max_frames, seed),
                                  'prefix_length': prefix.shape[1], 'stop_probabilities': stop}
                    record = {'image_id': image_id, 'level': level, 'seed': seed,
                              'oracle_text_used_offline': level in {'L0', 'L1', 'L2', 'L3'},
                              'image_only': level == 'L4', 'human_quality': 'unrated', **result}
                    results.append(record);print(json.dumps(record), flush=True)
        control_image = torch.zeros(1, 768, device=speaker.device)
        controls = [(k, semantic_condition(control_image, coefficient, control=k, training_mean=mean.to(speaker.device)))
                    for k in ['zero_image_through_adapter', 'zero_semantic_after_adapter', 'mean_semantic_baseline']]
        for image_id in args.development_ids:
            image = torch.tensor([by_image[image_id]['features']], device=speaker.device)
            controls.append(('development_' + image_id, semantic_condition(image, coefficient)))
        for i, image_id in enumerate(args.ids):
            other = args.ids[(i+1) % len(args.ids)]
            controls.append(('shuffled_' + image_id + '_from_' + other,
                             semantic_condition(torch.tensor([by_image[other]['features']], device=speaker.device), coefficient)))
        for name, semantic in controls:
            for seed in args.seeds:
                prefix, stop = model.sample(semantic, seed=seed, sample_modes=True)
                result = {'level': 'L5', 'control': name, 'seed': seed, 'semantic_norm': float(semantic.norm()),
                          'oracle_text_used_offline': name == 'mean_semantic_baseline', 'human_quality': 'unrated',
                          'prefix_length': prefix.shape[1], 'stop_probabilities': stop,
                          **render(builder, wrapper, prefix, speaker, args.output_dir / f'{name}-s{seed}.wav', args.max_frames, seed)}
                results.append(result);print(json.dumps(result), flush=True)
    metadata = {'scope': 'Training-oracle diagnostics plus exposed image-only development; no fresh access or training',
        'source_revision': os.environ.get('LD_CODE_REVISION'), 'model_revision': info['model_revision'],
        'model_id': info['model_id'], 'hashes': hashes, 'lock_sha256': digest(Path('uv.lock')),
        'source_sha256': digest(Path(__file__)), 'prior_diagnostics_sha256': digest(Path('src/latent_diaries/prior_diagnostics.py')),
        'semantic_adapter_sha256': digest(Path('src/latent_diaries/semantic_adapter.py')),
        'feature_provenance': evidence, 'seeds': args.seeds, 'max_frames': args.max_frames,
        'decoding': decoding, 'precision': 'float32', 'speaker_sha256': digest(Path(os.environ['LD_REF_AUDIO'])),
        'text_cache_manifest_sha256': digest(args.features_dir / 'manifest.json'),
        'adapter_manifest_sha256': digest(args.adapter_dir / 'manifest.json'), 'results': results}
    for key, path in paths.items():
        if digest(path) != hashes[key]:
            raise RuntimeError('Historical artifact changed: ' + key)
    (args.output_dir / 'results.json').write_text(json.dumps(metadata, indent=2) + '\n')
    (args.output_dir / 'prior-traces.json').write_text(json.dumps(traces) + '\n')
    (args.output_dir / 'completion.json').write_text(json.dumps({'clips': len(results), 'training_performed': False}) + '\n')


if __name__ == '__main__':
    main()
