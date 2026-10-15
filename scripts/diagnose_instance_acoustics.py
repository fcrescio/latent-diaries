"""Training-only oracle/matched/shuffled acoustic likelihood; never an art score."""
import argparse
import json
from pathlib import Path

import torch
from image_probe import make_reader
from safetensors.torch import load_file

from latent_diaries.inversion import condition, digest, load_backbone


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--teachers', type=Path, required=True)
    parser.add_argument('--baseline-run', type=Path, required=True)
    parser.add_argument('--acoustic-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not Path('/.dockerenv').exists() or args.output.exists():
        raise ValueError('Use managed worker and fresh output')
    rows = [r for r in json.loads(args.dataset.read_text())['rows'] if r['split'] == 'train']
    manifest = json.loads((args.teachers / 'manifest.json').read_text())
    if manifest['dataset_sha256'] != digest(args.dataset):
        raise ValueError('Teacher dataset differs')
    targets = {}
    for r in manifest['records']:
        for name in ['prefix', 'codes']:
            if digest(Path(r[f'{name}_file'])) != r[f'{name}_sha256']:
                raise ValueError('Teacher tensor changed')
        targets[str(Path(r['prefix_file']).parent)] = (
            load_file(r['prefix_file'])['soft_tokens'], load_file(r['codes_file'])['audio_codes'])
    info = json.loads((args.acoustic_run / 'config.json').read_text())
    wrapper, builder, speaker = load_backbone(info)
    result = {'scope': 'Training gold-previous-code likelihood only, no quality acceptance', 'methods': {}}
    with torch.inference_mode():
        for name, folder in [('baseline', args.baseline_run), ('acoustic', args.acoustic_run)]:
            config = json.loads((folder / 'config.json').read_text())
            weights = folder / 'reader.safetensors'
            if digest(weights) != config['reader_sha256'] or config['dataset_sha256'] != digest(args.dataset):
                raise ValueError('Reader/dataset differs')
            reader = make_reader(config, load_file(str(weights))).to(speaker.device).eval()
            prefixes = []
            for row in rows:
                prefix, _, lengths = reader(torch.tensor([row['features']], device=speaker.device))
                prefixes.append(prefix[:, :int(lengths[0])])
            metrics = []
            for i, row in enumerate(rows):
                oracle, gold = [v.to(speaker.device) for v in targets[row['cache_dir']]]
                record = {'id': row['id']}
                for kind, prefix in [('oracle', oracle), ('matched', prefixes[i]), ('shuffled', prefixes[(i+1) % len(rows)])]:
                    out = wrapper.model.talker(**builder.build_teacher_forced(condition(prefix, speaker), gold))
                    record[kind] = float(out.loss)
                metrics.append(record)
            result['methods'][name] = {'reader_sha256': digest(weights), 'rows': metrics,
                'mean_codec0_ce': {k: sum(r[k] for r in metrics)/len(metrics) for k in ['oracle', 'matched', 'shuffled']}}
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    args.output.chmod(0o600)
    print(json.dumps({k: v['mean_codec0_ce'] for k, v in result['methods'].items()}), flush=True)


if __name__ == '__main__':
    main()
