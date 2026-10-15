"""Generate training-supported acoustic targets, never read fresh/test images."""
import argparse
import json
from pathlib import Path

import soundfile as sf
import torch
from safetensors.torch import load_file, save_file

from latent_diaries.inversion import condition, digest, load_backbone


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if not Path('/.dockerenv').exists() or args.output_dir.exists():
        raise ValueError('Use managed worker and a new teacher directory')
    rows = json.loads(args.dataset.read_text())['rows']
    paths = sorted({r['cache_dir'] for r in rows if r['split'] == 'train'})
    info = json.loads((Path(paths[0]) / 'config.json').read_text())
    if info['lock_sha256'] != digest(Path('uv.lock')):
        raise ValueError('Teacher dependency lock differs')
    wrapper, builder, speaker = load_backbone(info)
    args.output_dir.mkdir(parents=True)
    records = []
    for index, value in enumerate(paths):
        path = Path(value) / 'prefix.safetensors'
        prefix = load_file(str(path))['soft_tokens'].to(speaker.device)
        with torch.inference_mode():
            codes = builder.generate_codes(condition(prefix, speaker), max_frames=160, seed=42)[0]
            if not 1 <= len(codes) < 160:
                raise ValueError('Teacher is empty or reaches duration cap; inspect before training')
            wavs, rate = wrapper.model.speech_tokenizer.decode([{'audio_codes': codes}])
        folder = args.output_dir / f'target-{index:02d}'
        folder.mkdir()
        save_file({'audio_codes': codes[None].cpu()}, str(folder / 'codes.safetensors'))
        sf.write(folder / 'positive.wav', wavs[0], rate)
        record = {'prefix_file': str(path), 'prefix_sha256': digest(path),
                  'codes_file': str(folder / 'codes.safetensors'),
                  'codes_sha256': digest(folder / 'codes.safetensors'),
                  'audio': str(folder / 'positive.wav'), 'audio_sha256': digest(folder / 'positive.wav'),
                  'frames': len(codes), 'codebooks': codes.shape[1], 'sample_rate': rate}
        records.append(record)
        print(json.dumps({'teacher': index + 1, 'frames': len(codes)}), flush=True)
    manifest = {k: info[k] for k in ['model_id', 'model_revision', 'lock_sha256', 'role_ids']}
    manifest.update(records=records, seed=42, max_frames=160, dataset_sha256=digest(args.dataset),
                    source_sha256=digest(Path(__file__)), linguistic_strings_read=False,
                    scope='Training-supported exact-prefix positive controls, not learned image outputs')
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
