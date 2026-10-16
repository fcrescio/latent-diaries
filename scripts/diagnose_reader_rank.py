"""Read-only training prefix rank bound; never reads development teacher targets."""
import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from latent_diaries.inversion import digest
from latent_diaries.reader_rank import reconstruction_rank_bound


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not Path('/.dockerenv').exists() or args.output.exists():
        raise ValueError('Managed worker and fresh diagnostic output required')
    info = json.loads((args.run_dir / 'config.json').read_text())
    weights = args.run_dir / 'reader.safetensors'
    if digest(weights) != info['reader_sha256'] or digest(args.dataset) != info['dataset_sha256']:
        raise ValueError('Reader/dataset differs')
    if info['architecture'] != 'global-mlp':
        raise ValueError('This bound assumes the global MLP linear output head')
    rows = json.loads(args.dataset.read_text())['rows']
    if any(r['split'] not in {'train', 'heldout'} for r in rows):
        raise ValueError('No fresh-test access')
    prefixes = []
    for row in rows:
        if row['split'] != 'train':
            continue
        path = Path(row['cache_dir']) / 'prefix.safetensors'
        if digest(path) != info['training_prefix_sha256'][row['id']]:
            raise ValueError('Training prefix changed')
        prefixes.append(load_file(str(path))['soft_tokens'][0])
    torch.set_num_threads(4)
    result = reconstruction_rank_bound(prefixes, load_file(str(weights))['scale'], info['width'])
    result.update(reader_sha256=digest(weights), source_sha256=digest(Path(__file__)),
                  bound_source_sha256=digest(Path('src/latent_diaries/reader_rank.py')))
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
