"""Offline Qwen contrast speech; all strings are diagnostic supervision only."""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from safetensors.torch import load_file
from scipy.signal import resample_poly

from latent_diaries.conditioning import VisualCondition
from latent_diaries.inversion import digest, load_backbone


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    if not Path('/.dockerenv').exists() or (args.output_dir.exists() and not args.resume):
        raise ValueError('Managed worker and fresh/resumable diagnostic targets required')
    benchmark = json.loads(args.benchmark.read_text())
    rows = benchmark['records']
    if len(rows) != 30 or any(r['split'] != 'train' or not r['offline_diagnostic_text_only'] for r in rows):
        raise ValueError('Frozen thirty existing training cases only')
    info = json.loads(Path('artifacts/m0_italian.json').read_text())
    wrapper, builder, speaker = load_backbone(info)
    metadata = {'scope':'Offline diagnostic minimal-pair and synthetic speaker-condition speech; no reader training',
        'benchmark_sha256':digest(args.benchmark), 'source_sha256':digest(Path(__file__)),
        'lock_sha256':digest(Path('uv.lock')), 'model_id':info['model_id'], 'model_revision':info['model_revision'],
        'source_revision':os.environ.get('LD_CODE_REVISION'), 'reference_audio_sha256':digest(Path(os.environ['LD_REF_AUDIO'])),
        'seed':42, 'batch_size':8, 'max_frames':160, 'codec0_greedy':False, 'residuals_sampled':True,
        'independent_real_speakers':False}
    args.output_dir.mkdir(mode=0o700, exist_ok=args.resume)
    manifest_path = args.output_dir/'manifest.json'
    old = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if old:
        for key,value in metadata.items():
            if key != 'source_revision' and old[key] != value:
                raise ValueError('Resume provenance differs: '+key)
    tasks, positives, alternate = [], {}, {}
    with torch.inference_mode():
        for row in rows:
            ids = wrapper._tokenize_texts([wrapper._build_assistant_text(row['source_text'])])[0][:,3:-5]
            prefix = wrapper.model.talker.text_projection(wrapper.model.talker.get_text_embeddings()(ids))
            if digest(Path(row['prefix_file'])) != row['prefix_sha256']:
                raise ValueError('Historical prefix changed')
            cached = load_file(row['prefix_file'])['soft_tokens'].to(speaker.device)
            torch.testing.assert_close(prefix, cached, rtol=0, atol=0)
            positives[row['id']] = prefix[0]
            for kind in ['attribute','negation']:
                text = row[kind+'_text']
                ids = wrapper._tokenize_texts([wrapper._build_assistant_text(text)])[0][:,3:-5]
                value = wrapper.model.talker.text_projection(wrapper.model.talker.get_text_embeddings()(ids))[0]
                tasks.append({'id':row['id']+'_'+kind,'source_id':row['id'],'kind':kind,
                              'text':text,'prefix':value,'speaker':speaker[0]})
        reference, sr = sf.read(os.environ['LD_REF_AUDIO'], dtype='float32')
        if reference.ndim == 2:
            reference = reference.mean(-1)
        chosen = [next(r for r in rows if r['scene']==scene) for scene in sorted({r['scene'] for r in rows})]
        for up, down in [(9,10),(11,10)]:
            shifted = resample_poly(reference,up,down).astype(np.float32)
            prompt = wrapper.create_voice_clone_prompt(ref_audio=(shifted,sr),x_vector_only_mode=True)
            vector = prompt[0].ref_spk_embedding.to(speaker.device)
            key = f'synthetic_reference_{up}_{down}'
            alternate[key] = {'source_reference_duration_factor':up/down,
                'speaker_vector_cosine_to_original':float(torch.nn.functional.cosine_similarity(vector[None],speaker)[0]),
                'speaker_vector_l2_change':float((vector-speaker[0]).norm()),
                'independent_person_or_real_speaker':False}
            for row in chosen:
                tasks.append({'id':row['id']+'_'+key,'source_id':row['id'],'kind':key,
                    'text':row['source_text'],'prefix':positives[row['id']],'speaker':vector})
        records = old['records'] if old else []
        by_id = {r['id']:r for r in records}
        for record in records:
            if digest(Path(record['audio'])) != record['audio_sha256']:
                raise ValueError('Cached contrast WAV changed')
        started = time.monotonic()
        for start in range(0,len(tasks),8):
            batch = tasks[start:start+8]
            if all(r['id'] in by_id for r in batch):
                continue
            if any(r['id'] in by_id for r in batch):
                raise ValueError('Partial batch cache; inspect rather than change batch RNG')
            values = [r['prefix'] for r in batch]
            lengths = torch.tensor([len(v) for v in values],device=speaker.device)
            prefixes = torch.nn.utils.rnn.pad_sequence(values,batch_first=True)
            mask = torch.arange(prefixes.shape[1],device=speaker.device)[None] < lengths[:,None]
            speakers = torch.stack([r['speaker'] for r in batch])
            generated = builder.generate_codes(VisualCondition(prefixes,mask,speakers),max_frames=160,seed=42+start)
            for row,codes in zip(batch,generated,strict=True):
                if not len(codes):
                    raise ValueError('Empty frozen-Qwen diagnostic output')
                wavs, rate = wrapper.model.speech_tokenizer.decode([{'audio_codes':codes}])
                path = args.output_dir/(row['id']+'.wav');sf.write(path,wavs[0],rate)
                record = {k:row[k] for k in ['id','source_id','kind','text']}
                record.update(audio=str(path),audio_sha256=digest(path),frames=len(codes),
                    reached_cap=len(codes)==160,sample_rate=rate,seconds=len(wavs[0])/rate,
                    seed=42+start,batch_offset=start,offline_text_used=True,human_quality='unrated')
                records.append(record);by_id[row['id']] = record
            result = {**metadata,'records':records,'synthetic_speaker_controls':alternate,
                'original_thirty_prefixes_match_historical_exactly':True,
                'elapsed_seconds':time.monotonic()-started,'peak_allocated_vram_bytes':torch.cuda.max_memory_allocated()}
            manifest_path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
            print(json.dumps({'generated':len(records),'total':len(tasks),'elapsed_seconds':result['elapsed_seconds']}),flush=True)
    print('Diagnostic targets complete; no optimization or neural training',flush=True)


if __name__ == '__main__':
    main()
