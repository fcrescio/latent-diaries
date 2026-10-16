"""Frozen image/audio scoring of bounded pools; no transcript, ASR or teacher reads."""
import argparse
import inspect
import json
import os
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

from latent_diaries.audio_semantic_feedback import select_audio_candidates
from latent_diaries.feature_provenance import verify_image_features
from latent_diaries.multimodal_feedback_encoder import FrozenFeedbackEncoder
from latent_diaries.vision import sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidates',type=Path,required=True)
    p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    args = p.parse_args()
    if not Path('/.dockerenv').exists() or args.output_dir.exists():
        raise ValueError('Managed worker and new no-training selection directory required')
    plan = json.loads(args.protocol.read_text());generated = json.loads(args.candidates.read_text())
    if generated['protocol_sha256'] != sha256(args.protocol):
        raise ValueError('Candidate protocol changed')
    if any(generated[k] for k in ['text_used','teacher_files_read','linguistic_labels_read']):
        raise ValueError('Runtime pools must be standalone image-only/no-target generation')
    rows = generated['results'];by_id = {r['id']:r for r in rows}
    ids = plan['development_ids'];shuffle = plan['shuffle']
    if set(by_id) != set(ids+['null_generation']) or sorted(shuffle)!=list(range(8)) or any(i==j for i,j in enumerate(shuffle)):
        raise ValueError('Frozen eight-case comparison and derangement required')
    evidence = verify_image_features([{k:r[k] for k in ['id','split','features']} for r in rows if r['id']!='null_generation'],
                                     generated['image_cache_folders'])
    for row in rows:
        if len(row['candidates']) != 4 or [c['seed'] for c in row['candidates']] != plan['seeds']:
            raise ValueError('Exactly four fixed candidates; no adaptive retries')
        for candidate in row['candidates']:
            if candidate.get('audio') and sha256(Path(candidate['audio'])) != candidate['audio_sha256']:
                raise ValueError('Candidate waveform changed')
    encoder = FrozenFeedbackEncoder('914f7f89142e33e77833254d9c9b90c3cef7303b','cuda',dtype='float32')
    args.output_dir.mkdir(mode=0o700)
    embeddings,keys = [],[]
    by_case = {}
    for row in rows:
        vector = torch.zeros(4,768)
        chosen = [i for i,c in enumerate(row['candidates']) if c['selection_eligible']]
        if chosen:
            encoded,_ = encoder.encode_audio([row['candidates'][i]['audio'] for i in chosen])
            vector[chosen] = encoded
        by_case[row['id']] = vector.numpy()
        embeddings.append(vector);keys.extend(row['id']+':'+str(c['seed']) for c in row['candidates'])
    feature_path = args.output_dir/'audio-features.safetensors'
    save_file({'embeddings':torch.cat(embeddings)},str(feature_path))
    results=[]
    for i,key in enumerate(ids):
        row = by_id[key]
        image = np.array(row['features'])
        wrong_image = np.array(by_id[ids[shuffle[i]]]['features'])
        eligible = [c['selection_eligible'] for c in row['candidates']]
        selection = select_audio_candidates(image,by_case[key],eligible,source_id=key,seed=plan['random_seed'],shuffled_image=wrong_image)
        null_row = by_id['null_generation']
        null_selection = select_audio_candidates(image,by_case['null_generation'],
            [c['selection_eligible'] for c in null_row['candidates']],source_id=key,seed=plan['random_seed'])
        methods={}
        for method,index in selection['indices'].items():
            methods[method] = {'candidate_index':index,'audio':None if index is None else row['candidates'][index]['audio'],
                'audio_sha256':None if index is None else row['candidates'][index]['audio_sha256'],
                'seed':None if index is None else row['candidates'][index]['seed'],
                'score':None if index is None else selection['image_audio_scores'][index],
                'human_quality':'unrated','quality_improvement_verified':False}
        result={'id':key,'methods':methods,'shuffled_scorer_image_id':ids[shuffle[i]],
            'candidate_image_audio_scores':selection['image_audio_scores'],'eligible_count':selection['eligible_count'],
            'fixed_random_draw':selection['fixed_random_draw'],
            'null_generation_selected_index':null_selection['indices']['image_audio_max'],
            'null_generation_image_audio_scores':null_selection['image_audio_scores'],
            'candidate_audio_pairwise_cosine':(by_case[key]@by_case[key].T).tolist()}
        results.append(result)
        print(json.dumps({'id':key,'selected_indices':selection['indices'],'eligible':selection['eligible_count']}),flush=True)
    output={'scope':'Phase2 bounded no-training audio-semantic selection; no quality/grounding promotion',
        'text_used':False,'teacher_files_read':False,'linguistic_labels_read':False,'asr_used':False,
        'source_revision':os.environ.get('LD_CODE_REVISION'),'source_sha256':sha256(Path(__file__)),
        'selector_source_sha256':sha256(Path(inspect.getfile(select_audio_candidates))),
        'encoder_source_sha256':sha256(Path(inspect.getfile(FrozenFeedbackEncoder))),
        'model_revision':encoder.revision,'vision_lock_sha256':sha256(Path('/opt/vision/uv.lock')),
        'worker_lock_sha256':sha256(Path('uv.lock')),'candidates_sha256':sha256(args.candidates),
        'protocol_sha256':sha256(args.protocol),'audio_features_sha256':sha256(feature_path),'audio_feature_row_ids':keys,
        'image_feature_provenance':evidence,'all_parameters_frozen':not any(p.requires_grad for p in encoder.model.parameters()),
        'peak_allocated_vram_bytes':torch.cuda.max_memory_allocated(),'results':results,
        'internal_score_gain_is_tautological_not_quality_evidence':True,'human_review':'pending',
        'neural_training':False,'temporal_reader_implemented':False,'reranker_promoted':False}
    (args.output_dir/'results.json').write_text(json.dumps(output,indent=2)+'\n')


if __name__ == '__main__':
    main()
