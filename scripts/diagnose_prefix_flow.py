"""Frozen prior/subspace diagnostics; supervision is read here only, not in inference."""
import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from train_prefix_flow import make_flow

from latent_diaries.inversion import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--corpus-dir', type=Path, required=True)
    parser.add_argument('--features-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not Path('/.dockerenv').exists() or args.output.exists():
        raise ValueError('Use managed worker and new diagnostic output')
    config = json.loads((args.run_dir / 'config.json').read_text())
    file = args.run_dir / 'reader.safetensors'
    assert digest(file) == config['reader_sha256']
    model = make_flow(load_file(str(file)), config['width']).eval().requires_grad_(False)
    corpus = json.loads((args.corpus_dir / 'corpus.json').read_text())['rows']
    prefix = load_file(str(args.corpus_dir / 'prefixes.safetensors'))['prefixes']
    features = load_file(str(args.features_dir / 'features.safetensors'))['embeddings']
    lengths = torch.tensor([r['length'] for r in corpus])
    with torch.inference_mode():
        recovered, recovered_lengths = model.unpack(model.pack(prefix, lengths))
        assert torch.equal(recovered_lengths, lengths)
        errors = (recovered - prefix).square().mean((1, 2))
        sampled, sampled_lengths = model.sample(features[:16], seed=42)
        changed, _ = model.sample(features[:16].roll(1, 0), seed=42)
    result = {'scope': 'Frozen subspace and numeric dependence diagnostics, not human quality',
              'oracle_reconstruction': {split: float(errors[torch.tensor([r['split'] == split for r in corpus])].mean())
                                        for split in ['train', 'heldout']},
              'sampled_prefix_finite': bool(torch.isfinite(sampled).all()),
              'sampled_length_range': [int(sampled_lengths.min()), int(sampled_lengths.max())],
              'same_noise_changed_features_mean_abs_difference': float((sampled - changed).abs().mean()),
              'reader_sha256': config['reader_sha256'], 'source_sha256': digest(Path(__file__))}
    assert digest(file) == config['reader_sha256']
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
