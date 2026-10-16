"""Training-only soft audio-code targets from the frozen matched Qwen teacher."""
import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from latent_diaries.inversion import digest, load_backbone, trajectory_logits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teachers', type=Path, required=True)
    parser.add_argument('--provenance-run', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if not Path('/.dockerenv').exists() or args.output_dir.exists():
        raise ValueError('Managed worker and fresh cache required')
    info = json.loads((args.provenance_run/'config.json').read_text())
    manifest = json.loads((args.teachers/'manifest.json').read_text())
    for key in ['model_id', 'model_revision', 'lock_sha256', 'dataset_sha256']:
        if info[key] != manifest[key]:
            raise ValueError('Teacher/backbone/dataset differs')
    wrapper, builder, speaker = load_backbone(info)
    args.output_dir.mkdir(parents=True)
    records = []
    with torch.inference_mode():
        for i, record in enumerate(manifest['records']):
            for kind in ['prefix', 'codes']:
                if digest(Path(record[kind+'_file'])) != record[kind+'_sha256']:
                    raise ValueError('Teacher input changed')
            prefix = load_file(record['prefix_file'])['soft_tokens'].to(speaker.device)
            gold = load_file(record['codes_file'])['audio_codes'].to(speaker.device)
            first, rest = trajectory_logits(wrapper, builder, prefix, speaker, gold)
            if not torch.isfinite(first).all() or not torch.isfinite(rest).all():
                raise ValueError('Nonfinite teacher distributions')
            target = args.output_dir/f'policy-{i:03d}.safetensors'
            save_file({'codec0_logits':first.cpu().contiguous(), 'residual_logits':rest.cpu().contiguous()}, str(target))
            records.append({'prefix_key':str(Path(record['prefix_file']).parent),
                            'codes_sha256':record['codes_sha256'], 'file':str(target),
                            'sha256':digest(target), 'frames':gold.shape[1]})
            if i % 20 == 0: print(json.dumps({'cached':i+1}),flush=True)
    output = {'records':records, 'teacher_manifest_sha256':digest(args.teachers/'manifest.json'),
              'lock_sha256':digest(Path('uv.lock')), 'source_sha256':digest(Path(__file__)),
              'inversion_source_sha256':digest(Path('src/latent_diaries/inversion.py')),
              'scope':'Training-only frozen teacher audio distributions; never inference inputs'}
    (args.output_dir/'manifest.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({'policies':len(records)}),flush=True)


if __name__ == '__main__':
    main()
