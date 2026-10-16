"""Generate training-supported acoustic targets, never read fresh/test images."""
import argparse
import json
from pathlib import Path

import soundfile as sf
import torch
from safetensors.torch import load_file, save_file

from latent_diaries.conditioning import VisualCondition
from latent_diaries.inversion import digest, load_backbone


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 8:
        raise ValueError('Teacher batch size must be1..8')
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
    for start in range(0, len(paths), args.batch_size):
        selected = paths[start:start + args.batch_size]
        prefixes = [load_file(str(Path(value) / 'prefix.safetensors'))['soft_tokens'][0].to(speaker.device)
                    for value in selected]
        lengths = torch.tensor([len(prefix) for prefix in prefixes], device=speaker.device)
        padded = torch.nn.utils.rnn.pad_sequence(prefixes, batch_first=True)
        mask = torch.arange(padded.shape[1], device=speaker.device)[None] < lengths[:, None]
        visual = VisualCondition(padded, mask, speaker.expand(len(selected), -1))
        with torch.inference_mode():
            generated = builder.generate_codes(visual, max_frames=160, seed=42)
        for offset, (value, codes) in enumerate(zip(selected, generated, strict=True)):
            index = start + offset
            path = Path(value) / 'prefix.safetensors'
            if not 1 <= len(codes) < 160:
                raise ValueError('Teacher is empty or reaches duration cap; inspect before training')
            with torch.inference_mode():
                wavs, rate = wrapper.model.speech_tokenizer.decode([{'audio_codes': codes}])
            folder = args.output_dir / f'target-{index:03d}'
            folder.mkdir()
            save_file({'audio_codes': codes[None].cpu()}, str(folder / 'codes.safetensors'))
            sf.write(folder / 'positive.wav', wavs[0], rate)
            record = {'prefix_file': str(path), 'prefix_sha256': digest(path),
                      'codes_file': str(folder / 'codes.safetensors'),
                      'codes_sha256': digest(folder / 'codes.safetensors'),
                      'audio': str(folder / 'positive.wav'), 'audio_sha256': digest(folder / 'positive.wav'),
                      'frames': len(codes), 'codebooks': codes.shape[1], 'sample_rate': rate,
                      'generation_batch_start': start, 'generation_batch_size': len(selected)}
            records.append(record)
            print(json.dumps({'teacher': index + 1, 'frames': len(codes)}), flush=True)
    manifest = {k: info[k] for k in ['model_id', 'model_revision', 'lock_sha256', 'role_ids']}
    manifest.update(records=records, seed=42, batch_size=args.batch_size, max_frames=160, dataset_sha256=digest(args.dataset),
                    source_sha256=digest(Path(__file__)), linguistic_strings_read=False,
                    scope='Training-supported exact-prefix positive controls, not learned image outputs')
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
