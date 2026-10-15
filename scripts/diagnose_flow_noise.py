"""One training-text diagnostic: noise versus length versus exact-prefix positive."""
import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from train_prefix_flow import make_flow

from latent_diaries.inversion import digest, load_backbone, render


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if not Path('/.dockerenv').exists() or args.output_dir.exists():
        raise ValueError('Use managed worker and new control directory')
    info = json.loads((args.root / 'reader/config.json').read_text())
    rows = json.loads((args.root / 'corpus/corpus.json').read_text())['rows']
    plan = json.loads((args.root / 'evaluation-plan.json').read_text())
    index = next(i for i, row in enumerate(rows) if row['id'] == plan['text_control_ids']['train'][0])
    features = load_file(str(args.root / 'features/features.safetensors'))['embeddings'][index:index + 1]
    target = load_file(str(args.root / 'corpus/prefixes.safetensors'))['prefixes'][index:index + 1]
    length = rows[index]['length']
    file = args.root / 'reader/reader.safetensors'
    assert digest(file) == info['reader_sha256']
    model = make_flow(load_file(str(file)), info['width'])
    wrapper, builder, speaker = load_backbone(info)
    model.to(speaker.device).eval().requires_grad_(False)
    features, target = features.to(speaker.device), target.to(speaker.device)
    args.output_dir.mkdir(parents=True)
    results = []
    for label, scale, oracle_length in [('unit_noise_gold_length', 1.0, True),
                                         ('quarter_noise', 0.25, False), ('zero_noise', 0.0, False)]:
        prefix, lengths = model.sample(features, seed=42, noise_scale=scale)
        n = length if oracle_length else int(lengths[0])
        result = render(builder, wrapper, prefix[:, :n], speaker, args.output_dir / f'{label}.wav', 105, 42)
        results.append({'id': label, 'noise_scale': scale, 'oracle_length': oracle_length,
                        'predicted_length': int(lengths[0]), 'gold_length': length,
                        'raw_prefix_mse': float((prefix[:, :length] - target[:, :length]).square().mean()), **result})
    with torch.inference_mode():
        recovered, _ = model.unpack(model.pack(target, torch.tensor([length], device=speaker.device)))
    results.append({'id': 'pca_oracle', **render(builder, wrapper, recovered[:, :length], speaker,
                                              args.output_dir / 'pca_oracle.wav', 105, 42)})
    results.append({'id': 'exact_positive', **render(builder, wrapper, target[:, :length], speaker,
                                                  args.output_dir / 'exact_positive.wav', 105, 42)})
    assert digest(file) == info['reader_sha256']
    result = {'scope': 'Training-text diagnostic with oracle controls; not learned image success',
              'source_sha256': digest(Path(__file__)), 'results': results, 'weights_updated': False}
    (args.output_dir / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'controls': len(results)}), flush=True)


if __name__ == '__main__':
    main()
