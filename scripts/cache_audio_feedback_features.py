"""Cache frozen media features and explicitly separate OFFLINE text references."""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path

import numpy as np
import torch
import transformers
from PIL import Image, ImageOps
from safetensors.torch import load_file, save_file
from scipy.signal import resample_poly

from latent_diaries.feature_provenance import checksum as sha256
from latent_diaries.multimodal_feedback_encoder import FrozenFeedbackEncoder, prepare_audio
from latent_diaries.vision import pool_global

VARIANTS = ['clean', 'attribute', 'negation', 'gain_half', 'noise20', 'speed09', 'speed11',
            'pad_half_second', 'reverse', 'shuffle100ms', 'phase_scramble', 'crop1s', 'crop2s', 'crop4s', 'last2s']


def waveform_variant(values, name, source_id):
    seed = int(hashlib.sha256((source_id+':'+name+':20261016').encode()).hexdigest()[:8],16)
    rng = np.random.default_rng(seed)
    if name == 'gain_half':
        return values*.5
    if name == 'noise20':
        noise = rng.normal(size=len(values));noise *= np.sqrt(np.mean(values.astype(np.float64)**2))/10
        return (values+noise).astype(np.float32)
    if name in {'speed09','speed11'}:
        return resample_poly(values,9 if name=='speed09' else 11,10).astype(np.float32)
    if name == 'pad_half_second':
        return np.pad(values,(8000,8000))
    if name == 'reverse':
        return np.ascontiguousarray(values[::-1])
    if name == 'shuffle100ms':
        chunks = [values[i:i+1600] for i in range(0,len(values),1600)]
        return np.concatenate([chunks[i] for i in rng.permutation(len(chunks))])
    if name == 'phase_scramble':
        spectrum = np.fft.rfft(values)
        phase = rng.uniform(-np.pi,np.pi,len(spectrum));phase[0] = phase[-1] = 0
        return np.fft.irfft(np.abs(spectrum)*np.exp(1j*phase),n=len(values)).astype(np.float32)
    if name.startswith('crop'):
        return values[:int(name[4])*16000]
    if name == 'last2s':
        return values[-32000:]
    raise ValueError('Unknown diagnostic waveform variant')


@torch.inference_mode()
def offline_text_reference(encoder, texts, format_name):
    """Diagnostic-only path. Never part of FrozenFeedbackEncoder public media API."""
    if format_name == 'Document':
        texts = ['title: none | text: '+t for t in texts]
    elif format_name == 'SentenceSimilarity':
        texts = ['task: sentence similarity | query: '+t for t in texts]
    elif format_name != 'unprefixed':
        raise ValueError('Unknown fixed diagnostic text format')
    inputs = encoder.processor(text=texts,return_tensors='pt',padding=True).to(encoder.device)
    output = encoder.model(**inputs)
    return pool_global(output.last_hidden_state,inputs['attention_mask']).cpu()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark', type=Path, required=True)
    p.add_argument('--targets', type=Path, required=True)
    p.add_argument('--human-export', type=Path, required=True)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    if not Path('/.dockerenv').exists() or (args.output_dir.exists() and not args.resume):
        raise ValueError('Managed worker and fresh/resumable feature cache required')
    benchmark = json.loads(args.benchmark.read_text())
    targets = json.loads(args.targets.read_text())
    if (benchmark['protocol_sha256'] != sha256(args.protocol)
            or targets['benchmark_sha256'] != sha256(args.benchmark)
            or benchmark['human_export_sha256'] != sha256(args.human_export)):
        raise ValueError('Frozen experiment provenance differs')
    rows = benchmark['records']
    if len(rows) != 30 or any(r['split'] != 'train' for r in rows):
        raise ValueError('Existing thirty-case training benchmark only')
    for r in rows:
        if sha256(r['image']) != r['image_sha256'] or sha256(r['audio']) != r['audio_sha256']:
            raise ValueError('Original benchmark media changed')
    for r in targets['records']:
        if sha256(r['audio']) != r['audio_sha256']:
            raise ValueError('Contrast waveform changed')
    args.output_dir.mkdir(mode=0o700,exist_ok=args.resume)
    manifest_path = args.output_dir/'manifest.json'
    info = {'scope':'Frozen training-only media feasibility; text arrays are offline reference features, never runtime inputs',
        'benchmark_sha256':sha256(args.benchmark),'targets_sha256':sha256(args.targets),
        'human_export_sha256':sha256(args.human_export),'protocol_sha256':sha256(args.protocol),
        'source_sha256':sha256(__file__),'encoder_source_sha256':sha256(Path(inspect.getfile(FrozenFeedbackEncoder))),
        'source_revision':os.environ.get('LD_CODE_REVISION'), 'model_revision':'914f7f89142e33e77833254d9c9b90c3cef7303b',
        'dtype':'float32','vision_lock_sha256':sha256('/opt/vision/uv.lock'),'worker_lock_sha256':sha256('uv.lock'),
        'torch_version':torch.__version__,'transformers_version':transformers.__version__,'batch_size':4,
        'row_ids':[r['id'] for r in rows],'seed':20261016,'neural_training':False}
    old = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if old:
        for key,value in info.items():
            if key != 'source_revision' and old[key] != value:
                raise ValueError('Feature resume differs: '+key)
    groups = old['groups'] if old else {}
    encoder = FrozenFeedbackEncoder(info['model_revision'],'cuda',dtype='float32')
    def cached(name, ids, callback):
        if name in groups:
            record = groups[name]
            if record['ids'] != ids or sha256(record['file']) != record['sha256']:
                raise ValueError('Completed cache group changed')
            return load_file(record['file'])['embeddings']
        collected, traces = [], []
        for start in range(0,len(ids),4):
            vector, trace = callback(start,min(start+4,len(ids)))
            collected.append(vector);traces.append(trace)
        values = torch.cat(collected).contiguous()
        if values.shape != (len(ids),768) or not torch.isfinite(values).all():
            raise ValueError('Invalid feature cache shape/value')
        path = args.output_dir/(name+'.safetensors')
        save_file({'embeddings':values},str(path))
        groups[name] = {'file':str(path),'sha256':sha256(path),'ids':ids,'shape':list(values.shape),'batch_traces':traces}
        output = {**info,'groups':groups,'all_parameters_frozen':not any(p.requires_grad for p in encoder.model.parameters()),
                  'peak_allocated_vram_bytes':torch.cuda.max_memory_allocated()}
        manifest_path.write_text(json.dumps(output,indent=2)+'\n')
        print(json.dumps({'completed_feature_group':name,'rows':len(ids)}),flush=True)
        return values
    identifiers = [r['id'] for r in rows]
    def images_for(items):
        values=[]
        for row in items:
            with Image.open(row['image']) as image:
                values.append(ImageOps.exif_transpose(image).convert('RGB'))
        return values
    visual = cached('image',identifiers,lambda a,b:encoder.encode_image(images_for(rows[a:b])))
    legacy = torch.tensor([r['features'] for r in rows])
    if float((visual-legacy).abs().max()) > 1e-4:
        raise ValueError('Full-media image path differs from historical pooled cache')
    all_texts = [r[k] for k in ['source_text','attribute_text','negation_text'] for r in rows]
    text_ids = [r['id']+'_'+kind for kind in ['source','attribute','negation'] for r in rows]
    for format_name in ['Document','SentenceSimilarity','unprefixed']:
        cached('text_'+format_name,text_ids,lambda a,b,f=format_name:(offline_text_reference(encoder,all_texts[a:b],f),
            {'offline_text_reference_only':True,'format':f,'variable_text_used':True}))
    base = [prepare_audio(r['audio'])[0] for r in rows]
    for variant in VARIANTS:
        if variant == 'clean':
            waves = base
        elif variant in {'attribute','negation'}:
            by_id = {r['source_id']:r for r in targets['records'] if r['kind']==variant}
            if set(by_id) != set(identifiers):
                raise ValueError('Incomplete contrast target set')
            waves = [prepare_audio(by_id[k]['audio'])[0] for k in identifiers]
        else:
            waves = [waveform_variant(wave,variant,key) for wave,key in zip(base,identifiers,strict=True)]
        cached('audio_'+variant,identifiers,lambda a,b,w=waves:encoder.encode_audio([(x,16000) for x in w[a:b]]))
    for kind in ['synthetic_reference_9_10','synthetic_reference_11_10']:
        records = [r for r in targets['records'] if r['kind']==kind]
        ids = [r['source_id'] for r in records]
        cached('audio_'+kind,ids,lambda a,b,rr=records:encoder.encode_audio([r['audio'] for r in rr[a:b]]))
    degraded = benchmark['human_rated_degraded']
    cached('audio_human_gibberish',[r['id'] for r in degraded],lambda a,b:encoder.encode_audio([r['audio'] for r in degraded[a:b]]))
    references = [r['mapping'] for r in json.loads(args.human_export.read_text()) if r['mapping']['level'] in {'L0','L1'}]
    cached('audio_exact_oracle_reference',[r['clip_id'] for r in references],lambda a,b:encoder.encode_audio([r['audio'] for r in references[a:b]]))
    same_scene = [i//3*3+(i%3+1)%3 for i in range(30)]
    other_scene = [(i+3)%30 for i in range(30)]
    for kind,image_order,audio_order in [('matched',list(range(30)),list(range(30))),
        ('same_scene_wrong_audio',list(range(30)),same_scene),
        ('other_scene_wrong_audio',list(range(30)),other_scene),
        ('same_audio_wrong_image',other_scene,list(range(30)))]:
        cached('joint_'+kind,identifiers,lambda a,b,ii=image_order,aa=audio_order:
            encoder.encode_image_audio(images_for([rows[i] for i in ii[a:b]]),[(base[i],16000) for i in aa[a:b]]))
    by_image = {r['id']:r for r in rows}
    cached('joint_human_gibberish',[r['id'] for r in degraded],lambda a,b:
        encoder.encode_image_audio(images_for([by_image[r['source_image_id']] for r in degraded[a:b]]),
                                   [r['audio'] for r in degraded[a:b]]))
    a, _ = encoder.encode_audio([rows[0]['audio']])
    pair, _ = encoder.encode_audio([rows[0]['audio'], rows[1]['audio']])
    b, _ = encoder.encode_audio([rows[1]['audio']])
    parity = max(float((a-pair[:1]).abs().max()),float((b-pair[1:]).abs().max()))
    output = json.loads(manifest_path.read_text());output['mixed_duration_batch_max_abs_error'] = parity
    output['image_cache_max_abs_error'] = float((visual-legacy).abs().max())
    output['all_feature_groups_complete'] = True
    manifest_path.write_text(json.dumps(output,indent=2)+'\n')
    if parity > 1e-4:
        raise ValueError('Mixed-duration audio parity failed; investigate padding before using batch features')
    print('Media/text-reference caches complete; no neural training',flush=True)


if __name__ == '__main__':
    main()
