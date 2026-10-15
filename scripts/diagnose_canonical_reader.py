"""Frozen development comparison; cached targets are diagnostic supervision only."""
import argparse
import json
from pathlib import Path

import torch
from image_probe import make_reader
from safetensors.torch import load_file

from latent_diaries.inversion import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--targets-dir', type=Path, required=True)
    parser.add_argument('--old-run', type=Path, required=True)
    parser.add_argument('--expanded-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not Path('/.dockerenv').exists() or args.output.exists():
        raise ValueError('Use managed worker and a new diagnostic output')
    summary, records, fingerprints = {}, {}, {}
    development_ids = None
    for cohort, folder in [('old', args.old_run), ('expanded', args.expanded_run)]:
        dataset_path = args.targets_dir / f'dataset-{cohort}.json'
        dataset = json.loads(dataset_path.read_text())
        config = json.loads((folder / 'config.json').read_text())
        path = folder / 'reader.safetensors'
        fingerprints[cohort] = digest(path)
        if fingerprints[cohort] != config['reader_sha256'] or digest(dataset_path) != config['dataset_sha256']:
            raise ValueError('Frozen checkpoint or dataset changed')
        if not config['fixed_length'] or config['evaluation_scope'] != 'development':
            raise ValueError('Expected fixed-length development model')
        rows = dataset['rows']
        ids = [r['id'] for r in rows if r['split'] == 'heldout']
        if development_ids is not None and ids != development_ids:
            raise ValueError('Development comparison differs')
        development_ids = ids
        model = make_reader(config, load_file(str(path))).eval().requires_grad_(False)
        with torch.inference_mode():
            predictions, masks, lengths = model(torch.tensor([r['features'] for r in rows]))
            measured = []
            for i, row in enumerate(rows):
                target = load_file(str(Path(row['cache_dir']) / 'prefix.safetensors'))['soft_tokens'][0]
                if int(lengths[i]) != len(target) or not bool(masks[i].all()):
                    raise ValueError('Constant prefix length parity failed')
                measured.append({'id': row['id'], 'split': row['split'],
                                 'raw_prefix_mse': float((predictions[i] - target).square().mean())})
        summary[cohort] = {}
        for split in ['train', 'heldout']:
            subset = [r for r in measured if r['split'] == split]
            summary[cohort][split] = {'count': len(subset), 'mean_raw_prefix_mse':
                                    sum(r['raw_prefix_mse'] for r in subset) / len(subset)}
        records[cohort] = measured
        if digest(path) != fingerprints[cohort]:
            raise ValueError('Diagnostic changed weights')
    threshold = 1e-5
    passed = summary['expanded']['heldout']['mean_raw_prefix_mse'] < threshold
    result = {'scope': 'Development only; no fresh-test access', 'summary': summary,
              'records': records, 'reader_sha256': fingerprints, 'weights_updated': False,
              'development_gate_passed': passed, 'gate_threshold': threshold,
              'human_quality_acceptance': False, 'source_sha256': digest(Path(__file__))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    args.output.chmod(0o600)
    print(json.dumps({k: result[k] for k in ['summary', 'development_gate_passed']}), flush=True)


if __name__ == '__main__':
    main()
