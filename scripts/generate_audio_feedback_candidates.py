"""Image-only bounded candidate pools for a no-training feedback comparison."""
import argparse
import inspect
import json
import os
from pathlib import Path

import torch
from evaluate_autoregressive_prior import load_prior
from safetensors.torch import load_file

from latent_diaries.feature_provenance import verify_image_features
from latent_diaries.image_reader import validate_image_inputs
from latent_diaries.inversion import digest, load_backbone, render
from latent_diaries.multihypothesis_prior import MultihypothesisPrior
from latent_diaries.semantic_adapter import semantic_condition


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--image-cache',type=Path,action='append',required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    args = p.parse_args()
    if not Path('/.dockerenv').exists() or args.output_dir.exists():
        raise ValueError('Managed worker and fresh candidate directory required')
    plan = json.loads(args.protocol.read_text())
    rows = json.loads(args.inputs.read_text())['rows']
    validate_image_inputs(rows,[r['id'] for r in rows],allow_new=True)
    if any(r['split'] not in {'train','heldout'} for r in rows):
        raise ValueError('Reserved/new input forbidden')
    evidence = verify_image_features(rows,args.image_cache)
    by_id = {r['id']:r for r in rows}
    if len(plan['development_ids']) != 8 or any(by_id[k]['split']!='heldout' for k in plan['development_ids']):
        raise ValueError('Eight frozen existing development inputs required')
    if plan['seeds'] != [42,41,43,44]:
        raise ValueError('Do not retune the candidate seed set')
    run = Path(plan['reader_run']);adapter_file = Path(plan['adapter'])
    info = json.loads((run/'config.json').read_text())
    adapter_info = json.loads((adapter_file.parent/'manifest.json').read_text())
    if digest(run/'reader.safetensors') != info['reader_sha256'] or digest(Path('uv.lock')) != info['lock_sha256']:
        raise ValueError('Prior or runtime changed')
    if (digest(adapter_file) != adapter_info['adapter_sha256'] or
        any(info[k] != adapter_info[k] for k in ['corpus_sha256','features_sha256'])):
        raise ValueError('Adapter/prior provenance differs')
    wrapper,builder,speaker = load_backbone(info)
    model = load_prior(load_file(str(run/'reader.safetensors')),info['recipe'],info['modes']).to(speaker.device).eval().requires_grad_(False)
    adapter = load_file(str(adapter_file))['coefficient'].to(speaker.device)
    args.output_dir.mkdir(mode=0o700)
    chosen = [by_id[k] for k in plan['development_ids']]+[{'id':'null_generation','split':'control','features':[0.]*768}]
    results = []
    for row in chosen:
        image = torch.tensor([row['features']],device=speaker.device)
        control = 'zero_semantic_after_adapter' if row['id']=='null_generation' else 'image'
        semantic = semantic_condition(image,adapter,control=control)
        records = []
        for seed in plan['seeds']:
            prefix,stop = model.sample(semantic,seed=seed,sample_modes=True)
            result = render(builder,wrapper,prefix,speaker,args.output_dir/f"{row['id']}-s{seed}.wav",plan['max_frames'],seed)
            eligible = bool(result.get('audio')) and result['frames'] < plan['max_frames']
            record = {'seed':seed,'prefix_length':prefix.shape[1],'prefix_stop_probabilities':stop,
                      'selection_eligible':eligible,'ineligible_reason':None if eligible else 'empty_or_frame_cap',**result}
            records.append(record)
            print(json.dumps({'image_id':row['id'],'seed':seed,'frames':result['frames'],'eligible':eligible}),flush=True)
        results.append({'id':row['id'],'split':row['split'],'features':row['features'],'candidates':records})
        output = {'scope':'Strict image-only candidate generation plus explicit semantic-null pool; no text or teacher files',
            'text_used':False,'teacher_files_read':False,'linguistic_labels_read':False,
            'source_revision':os.environ.get('LD_CODE_REVISION'),'source_sha256':digest(Path(__file__)),
            'prior_source_sha256':digest(Path(inspect.getfile(MultihypothesisPrior))),
            'reader_sha256':info['reader_sha256'],'adapter_sha256':adapter_info['adapter_sha256'],
            'model_id':info['model_id'],'model_revision':info['model_revision'],'worker_lock_sha256':digest(Path('uv.lock')),
            'inputs_sha256':digest(args.inputs),'protocol_sha256':digest(args.protocol),'feature_provenance':evidence,
            'image_cache_folders':[str(p) for p in args.image_cache],'results':results,
            'human_quality':'unrated','reranker_promotion':False}
        (args.output_dir/'results.json').write_text(json.dumps(output,indent=2)+'\n')
    if digest(run/'reader.safetensors') != info['reader_sha256'] or digest(adapter_file) != adapter_info['adapter_sha256']:
        raise RuntimeError('Historical weights changed')


if __name__ == '__main__':
    main()
