"""One failing teacher fragment: isolate official single-item greedy versus stock sampling."""
import argparse
import json
import os
from pathlib import Path

import soundfile as sf
import torch
from safetensors.torch import save_file

from latent_diaries.inversion import digest, load_backbone


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus', type=Path, required=True)
    p.add_argument('--sequence-id', required=True)
    p.add_argument('--fragment-index', type=int, choices=[1,2], required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    if not Path('/.dockerenv').exists() or args.output_dir.exists():
        raise ValueError('Managed worker and fresh diagnostic output required')
    corpus = json.loads(args.corpus.read_text())
    row = next(r for r in corpus['rows'] if r['id'] == args.sequence_id)
    text = row['fragments'][args.fragment_index-1]
    args.output_dir.mkdir(mode=0o700)
    wrapper, _, speaker = load_backbone(json.loads(Path('artifacts/m0_italian.json').read_text()))
    prompt = wrapper.create_voice_clone_prompt(ref_audio=os.environ['LD_REF_AUDIO'], x_vector_only_mode=True)
    results = []
    for greedy in [True, False]:
        captured = {};original = wrapper.model.generate
        def capture(*values, _original=original, _captured=captured, **kwargs):
            output = _original(*values, **kwargs);_captured['codes'] = output[0][0].detach().cpu()
            return output
        wrapper.model.generate = capture
        try:
            with torch.inference_mode(), torch.random.fork_rng(devices=[speaker.device.index or 0]):
                torch.manual_seed(42)
                waves, rate = wrapper.generate_voice_clone(text=text, language='Italian', voice_clone_prompt=prompt,
                    non_streaming_mode=True, max_new_tokens=241, min_new_tokens=2, do_sample=not greedy,
                    top_k=50, top_p=1.0, temperature=.9, repetition_penalty=1.05,
                    subtalker_dosample=True, subtalker_top_k=50, subtalker_top_p=1.0, subtalker_temperature=.9)
        finally:
            wrapper.model.generate = original
        name = 'single_greedy' if greedy else 'single_sampled'
        code = captured['codes'];file = args.output_dir/(name+'.wav');sf.write(file, waves[0], rate)
        save_file({'audio_codes': code[None]}, str(args.output_dir/(name+'.safetensors')))
        result = {'mode': name, 'frames': len(code), 'shape': list(code.shape),
                  'seconds': len(waves[0])/rate, 'reached_cap': len(code)>=240,
                  'audio': str(file), 'audio_sha256': digest(file)}
        results.append(result);print(json.dumps(result), flush=True)
    (args.output_dir/'report.json').write_text(json.dumps({'scope': 'Two isolated official-teacher decoding controls, not candidate selection',
        'corpus_sha256': digest(args.corpus), 'source_sha256': digest(Path(__file__)),
        'sequence_id': args.sequence_id, 'fragment_index': args.fragment_index, 'seed': 42,
        'results': results}, indent=2)+'\n')


if __name__ == '__main__':
    main()
