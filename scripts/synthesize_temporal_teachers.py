"""Official text-conditioned Qwen teacher upper bound; no reader or training."""
import argparse
import inspect
import json
import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from safetensors.torch import save_file

from latent_diaries.inversion import digest, load_backbone


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['corpus', 'protocol', 'output-dir']:
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--batch-size', type=int, choices=[1,4], default=4)
    p.add_argument('--codec0-sampling', action='store_true', help='Use official stock first-codebook sampling globally, never per-case retries')
    p.add_argument('--fixed-seed', action='store_true', help='Restart the same baseline seed for every isolated utterance')
    args = p.parse_args()
    if not Path('/.dockerenv').exists() or (args.output_dir.exists() and not args.resume):
        raise ValueError('Managed worker and fresh/resumable teacher output required')
    if args.fixed_seed and args.batch_size != 1:
        raise ValueError('Fixed per-utterance seed requires isolated batch size1')
    corpus = json.loads(args.corpus.read_text());protocol = json.loads(args.protocol.read_text())
    if corpus['protocol_sha256'] != digest(args.protocol) or len(corpus['rows']) != 24:
        raise ValueError('Frozen24-sequence corpus required')
    info = json.loads(Path('artifacts/m0_italian.json').read_text())
    if any(info[k] != protocol['teacher'][k2] for k,k2 in [('model_id','model_id'),('model_revision','revision')]):
        raise ValueError('Pinned teacher differs')
    args.output_dir.mkdir(mode=0o700, exist_ok=args.resume)
    meta = {'scope': 'Official text-conditioned teacher baseline, never learned textless output',
        'corpus_sha256': digest(args.corpus), 'protocol_sha256': digest(args.protocol),
        'source_sha256': digest(Path(__file__)), 'worker_lock_sha256': digest(Path('uv.lock')),
        'source_revision': os.environ.get('LD_CODE_REVISION'), 'model_id': info['model_id'],
        'model_revision': info['model_revision'], 'reference_sha256': digest(Path(os.environ['LD_REF_AUDIO'])),
        'batch_size': args.batch_size, 'seed': 42, 'max_frames': protocol['teacher']['max_frames'],
        'fixed_per_utterance_seed': args.fixed_seed,
        'codec0_greedy': not args.codec0_sampling, 'residuals_sampled': True, 'text_used_offline': True,
        'reference_transcript_used': False, 'training_started': False, 'human_validated': False}
    path = args.output_dir/'manifest.json';old = json.loads(path.read_text()) if path.exists() else None
    if old:
        for key, value in meta.items():
            if key != 'source_revision' and old[key] != value:
                raise ValueError('Changed teacher resume provenance: '+key)
    records = old['records'] if old else []
    for record in records:
        if record['reached_cap']:
            raise ValueError('Failed teacher batch requires diagnosis, never an adaptive resume')
        if digest(Path(record['audio'])) != record['audio_sha256'] or digest(Path(record['codes'])) != record['codes_sha256']:
            raise ValueError('Completed teacher artifact changed')
    wrapper, _, speaker = load_backbone(info)
    meta['upstream_wrapper_source_sha256'] = digest(Path(inspect.getfile(type(wrapper))))
    if old and old['upstream_wrapper_source_sha256'] != meta['upstream_wrapper_source_sha256']:
        raise ValueError('Installed upstream wrapper changed')
    voice_prompt = wrapper.create_voice_clone_prompt(ref_audio=os.environ['LD_REF_AUDIO'], x_vector_only_mode=True)
    torch.testing.assert_close(voice_prompt[0].ref_spk_embedding.to(speaker.device), speaker[0], rtol=0, atol=0)
    tasks = [{'id': r['id']+f'-f{i+1}', 'sequence_id': r['id'], 'fragment_index': i+1,
              'text': text, 'split': r['split']} for r in corpus['rows'] for i, text in enumerate(r['fragments'])]
    complete = {r['id'] for r in records};started = time.monotonic()
    with torch.inference_mode():
        for start in range(0, len(tasks), args.batch_size):
            batch = tasks[start:start+args.batch_size]
            if all(r['id'] in complete for r in batch):
                continue
            if any(r['id'] in complete for r in batch):
                raise ValueError('Partial completed batch; preserve RNG and inspect')
            captured = {};original = wrapper.model.generate
            seed = 42 if args.fixed_seed else 42+start
            def capture_generate(*values, _original=original, _captured=captured, **kwargs):
                output = _original(*values, **kwargs)
                _captured['codes'] = [c.detach().cpu() for c in output[0]]
                return output
            wrapper.model.generate = capture_generate
            try:
                with torch.random.fork_rng(devices=[speaker.device.index or 0]):
                    torch.manual_seed(seed)
                    wavs, rate = wrapper.generate_voice_clone(text=[r['text'] for r in batch], language='Italian',
                        voice_clone_prompt=voice_prompt, non_streaming_mode=True,
                        do_sample=args.codec0_sampling, top_k=50, top_p=1.0, temperature=.9,
                        repetition_penalty=1.05, subtalker_dosample=True,
                        subtalker_top_k=50, subtalker_top_p=1.0, subtalker_temperature=.9,
                        max_new_tokens=meta['max_frames']+1, min_new_tokens=2)
            finally:
                wrapper.model.generate = original
            codes = captured['codes']
            if not len(wavs) == len(codes) == len(batch):
                raise ValueError('Official batch boundaries changed')
            for task, wave, code in zip(batch, wavs, codes, strict=True):
                if code.ndim != 2 or code.shape[1] != 16 or len(code) < 1:
                    raise ValueError('Invalid/empty teacher codes; inspect before proceeding')
                if rate != 24000 or not np.isfinite(wave).all():
                    raise ValueError('Genuine finite24kHz teacher waveform required')
                folder = args.output_dir/task['id'];folder.mkdir()
                audio, codefile = folder/'teacher.wav', folder/'codes.safetensors'
                sf.write(audio, wave, rate);save_file({'audio_codes': code[None]}, str(codefile))
                records.append({k: task[k] for k in ['id','sequence_id','fragment_index','split']} | {
                    'audio': str(audio), 'audio_sha256': digest(audio), 'codes': str(codefile),
                    'codes_sha256': digest(codefile), 'frames': len(code), 'codebooks': 16,
                    'seconds': len(wave)/rate, 'sample_rate': rate, 'seed': seed,
                    'batch_start': start, 'batch_size': len(batch), 'reached_cap': len(code)>=meta['max_frames']})
            path.write_text(json.dumps({**meta, 'records': records, 'elapsed_seconds': time.monotonic()-started,
                'peak_vram_bytes': torch.cuda.max_memory_allocated()}, indent=2)+'\n')
            print(json.dumps({'utterances': len(records), 'total': 48}), flush=True)
            if any(r['reached_cap'] for r in records):
                raise ValueError('Frame-capped teacher batch retained with all results; diagnose before further synthesis')
    print('Official teacher baseline complete; human corpus validation still pending', flush=True)


if __name__ == '__main__':
    main()
