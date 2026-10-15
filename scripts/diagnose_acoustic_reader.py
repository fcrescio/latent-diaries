"""Frozen teacher-forced development diagnostics, separate from textless rendering."""
import argparse
import json
from pathlib import Path

import torch
from image_probe import make_reader
from safetensors.torch import load_file

from latent_diaries.acoustic_reader import acoustic_objective
from latent_diaries.inversion import digest, load_backbone


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--teachers', type=Path, required=True)
    parser.add_argument('--baseline-run', type=Path, required=True)
    parser.add_argument('--acoustic-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not Path('/.dockerenv').exists() or args.output.exists():
        raise ValueError('Use managed worker and fresh diagnostic output')
    rows = json.loads(args.dataset.read_text())['rows']
    if any(r['split'] not in {'train', 'heldout'} for r in rows):
        raise ValueError('Development analysis only')
    # One fixed training control per target, all existing development rows.
    seen, selected = set(), []
    for row in sorted(rows, key=lambda r: r['id']):
        if row['split'] == 'heldout' or row['cache_dir'] not in seen:
            selected.append(row)
        if row['split'] == 'train':
            seen.add(row['cache_dir'])
    info = json.loads((args.acoustic_run / 'config.json').read_text())
    wrapper, builder, speaker = load_backbone(info)
    manifest = args.teachers / 'manifest.json'
    if digest(manifest) != info['teacher_manifest_sha256']:
        raise ValueError('Teacher manifest changed')
    targets = {}
    for record in json.loads(manifest.read_text())['records']:
        for kind in ['prefix', 'codes']:
            if digest(Path(record[kind + '_file'])) != record[kind + '_sha256']:
                raise ValueError('Teacher tensors changed')
        targets[str(Path(record['prefix_file']).parent)] = (
            load_file(record['prefix_file'])['soft_tokens'].to(speaker.device),
            load_file(record['codes_file'])['audio_codes'].to(speaker.device))
    summary, all_metrics, fingerprints = {}, {}, {}
    for method, folder in [('baseline', args.baseline_run), ('acoustic', args.acoustic_run)]:
        config = json.loads((folder / 'config.json').read_text())
        path = folder / 'reader.safetensors'
        fingerprints[method] = digest(path)
        if config['reader_sha256'] != fingerprints[method] or config['dataset_sha256'] != digest(args.dataset):
            raise ValueError('Reader/dataset changed')
        reader = make_reader(config, load_file(str(path))).to(speaker.device).eval().requires_grad_(False)
        measured = []
        with torch.inference_mode():
            for row in selected:
                target, gold = targets[row['cache_dir']]
                prefix = reader(torch.tensor([row['features']], device=speaker.device))[0]
                loss, values = acoustic_objective(wrapper, builder, prefix, speaker, gold, target, reader.scale)
                measured.append({'id': row['id'], 'split': row['split'], 'total_loss': float(loss),
                                 **{k: float(v) for k, v in values.items()},
                                 'raw_prefix_mse': float((prefix - target).square().mean())})
        summary[method] = {}
        for split in ['train', 'heldout']:
            subset = [r for r in measured if r['split'] == split]
            summary[method][split] = {'count': len(subset), **{
                key: sum(r[key] for r in subset) / len(subset)
                for key in ['total_loss', 'codec0_ce', 'residual_ce', 'normalized_prefix_mse', 'raw_prefix_mse']}}
        all_metrics[method] = measured
        if digest(path) != fingerprints[method]:
            raise ValueError('Diagnostic mutated weights')
    result = {'scope': 'Exposed-development teacher forcing; not autoregressive or human acceptance',
              'summary': summary, 'metrics': all_metrics, 'reader_sha256': fingerprints,
              'weights_updated': False, 'human_quality_acceptance': False,
              'source_sha256': digest(Path(__file__))}
    args.output.write_text(json.dumps(result, indent=2) + '\n');args.output.chmod(0o600)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
