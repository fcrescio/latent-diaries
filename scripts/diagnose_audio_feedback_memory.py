"""Order/nonspeech probes only; concatenated controls are NOT narrative training data."""
import argparse
import inspect
import json
import os
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

from latent_diaries.audio_feedback_metrics import cluster_mean_interval
from latent_diaries.feature_provenance import checksum
from latent_diaries.multimodal_feedback_encoder import FrozenFeedbackEncoder, prepare_audio


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    args = p.parse_args()
    if not Path('/.dockerenv').exists() or args.output_dir.exists():
        raise ValueError('Managed worker and new diagnostic directory required')
    rows = json.loads(args.inputs.read_text())['rows']
    if len(rows)!=30 or any(set(r)!={'id','audio','audio_sha256'} for r in rows):
        raise ValueError('Thirty frozen media-only inputs required')
    for r in rows:
        if checksum(r['audio'])!=r['audio_sha256']:
            raise ValueError('Audio changed')
    audio = [prepare_audio(r['audio'])[0] for r in rows]
    partner = [i//3*3+(i%3+1)%3 for i in range(30)]
    pause = np.zeros(4800,dtype=np.float32)
    variants = {'single':audio,
        'AB':[np.concatenate([audio[i],pause,audio[j]]) for i,j in enumerate(partner)],
        'BA':[np.concatenate([audio[j],pause,audio[i]]) for i,j in enumerate(partner)],
        'AA':[np.concatenate([a,pause,a]) for a in audio]}
    if any(len(w)>480000 for waves in variants.values() for w in waves):
        raise ValueError('Never truncate long sequence controls')
    encoder = FrozenFeedbackEncoder('914f7f89142e33e77833254d9c9b90c3cef7303b','cuda')
    values = {}
    for key,waves in variants.items():
        batches = [encoder.encode_audio([(a,16000) for a in waves[i:i+4]])[0] for i in range(0,len(waves),4)]
        values[key] = torch.cat(batches)
        print(json.dumps({'order_control':key,'rows':len(waves)}),flush=True)
    a = values['single'];b = a[partner]
    ab,ba,aa = [values[k] for k in ['AB','BA','AA']]
    forward = ((ab*b).sum(-1)-(ab*a).sum(-1)).numpy()
    backward = ((ba*a).sum(-1)-(ba*b).sum(-1)).numpy()
    ids = [r['id'] for r in rows]
    metrics = {'AB_BA_cosine':cluster_mean_interval((ab*ba).sum(-1).numpy(),ids),
        'AB_last_minus_first_similarity':cluster_mean_interval(forward,ids),
        'BA_last_minus_first_similarity':cluster_mean_interval(backward,ids),
        'last_segment_preference_balanced':cluster_mean_interval(((forward>0).astype(float)+(backward>0).astype(float))/2,ids),
        'AA_single_cosine':cluster_mean_interval((aa*a).sum(-1).numpy(),ids)}
    t = np.arange(80000,dtype=np.float64)/16000
    rng = np.random.default_rng(20261016)
    controls = [(.04*np.sin(2*np.pi*440*t)).astype(np.float32),
                rng.normal(0,.04,size=len(t)).astype(np.float32),
                (.04*np.sin(2*np.pi*(220*t+110*t*t))).astype(np.float32)]
    noise,noise_trace = encoder.encode_audio([(c,16000) for c in controls])
    silence_rejected = False
    try:
        encoder.encode_audio([(np.zeros(16000,dtype=np.float32),16000)])
    except ValueError:
        silence_rejected = True
    args.output_dir.mkdir(mode=0o700)
    path = args.output_dir/'features.safetensors'
    save_file({**values,'nonspeech_controls':noise},str(path))
    output = {'scope':'Media-order and clearly labeled nonspeech controls; not coherent continuations or speech training data',
        'input_sha256':checksum(args.inputs),'protocol_sha256':checksum(args.protocol),
        'source_revision':os.environ.get('LD_CODE_REVISION'),'source_sha256':checksum(__file__),
        'encoder_source_sha256':checksum(inspect.getfile(FrozenFeedbackEncoder)),
        'model_revision':encoder.revision,'dtype':'float32','vision_lock_sha256':checksum('/opt/vision/uv.lock'),
        'worker_lock_sha256':checksum('uv.lock'),'features_sha256':checksum(path),'metrics':metrics,
        'audio_ids':ids,'partner_indices':partner,'pause_seconds':.3,
        'nonspeech_control_labels':['sine440Hz','white_noise','chirp'],
        'nonspeech_trace':noise_trace,'silence_rejected':silence_rejected,
        'nonspeech_similarity_to_teacher_audio':(noise@a.T).tolist(),
        'variable_text_used':False,'neural_training':False,'narrative_progression_proven':False}
    (args.output_dir/'report.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(metrics,indent=2),flush=True)


if __name__ == '__main__':
    main()
