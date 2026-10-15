"""Build canonical target arrays in worker; no scene text enters inference specs."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from latent_diaries.inversion import digest, load_backbone


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--design',type=Path,required=True)
    parser.add_argument('--canonical-texts',type=Path,required=True)
    parser.add_argument('--features-dir',type=Path,action='append',required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    if not Path('/.dockerenv').exists() or args.output_dir.exists():
        raise ValueError('Use managed worker with new target directory')
    design=json.loads(args.design.read_text());canonical=json.loads(args.canonical_texts.read_text())
    by_id={};feature_hashes=[];manifest_hashes=[]
    for folder in args.features_dir:
        manifest=json.loads((folder/'manifest.json').read_text());path=folder/'features.safetensors'
        if digest(path)!=manifest['features_sha256']:raise ValueError('Feature checksum differs')
        values=load_file(str(path))['embeddings']
        for i,row in enumerate(manifest['records']):
            if row['id'] in by_id:raise ValueError('Duplicate feature identities')
            by_id[row['id']]=values[i].tolist()
        feature_hashes.append(digest(path));manifest_hashes.append(digest(folder/'manifest.json'))
    info=json.loads(Path('artifacts/m0_italian.json').read_text())
    wrapper,builder,_=load_backbone(info)
    args.output_dir.mkdir(parents=True)
    for scene in design['active_classes']:
        text=canonical['texts'][scene]
        ids=wrapper._tokenize_texts([wrapper._build_assistant_text(text)])[0]
        with torch.no_grad():
            prefix=wrapper.model.talker.text_projection(wrapper.model.talker.get_text_embeddings()(ids[:,3:-5]))
        if prefix.shape[1]!=canonical['prefix_length']:raise ValueError('Full-wrapper/tokenizer length parity failed')
        folder=args.output_dir/'class-targets'/scene;folder.mkdir(parents=True)
        save_file({'soft_tokens':prefix.cpu()},str(folder/'prefix.safetensors'))
        metadata={k:info[k] for k in ['model_id','model_revision']}
        metadata.update(lock_sha256=digest(Path('uv.lock')),role_ids=list(builder.role_ids))
        (folder/'config.json').write_text(json.dumps(metadata,indent=2)+'\n')
    for cohort in ['old','expanded']:
        entries=[]
        for row in design['rows']:
            if row['split']=='train' and cohort=='old' and row['cohort']!='old':continue
            entries.append({k:row[k] for k in ['id','split','scene_group']} | {
                'features':by_id[row['id']], 'cache_dir':str(args.output_dir/'class-targets'/row['scene'])})
        metadata={'rows':entries,'design_sha256':digest(args.design),
            'image_features_sha256':hashlib.sha256('|'.join(feature_hashes).encode()).hexdigest(),
            'vision_manifest_sha256':hashlib.sha256('|'.join(manifest_hashes).encode()).hexdigest(),
            'scope':'canonical development; fresh cases not accessed',
            'canonical_text_sha256':digest(args.canonical_texts)}
        (args.output_dir/f'dataset-{cohort}.json').write_text(json.dumps(metadata,indent=2)+'\n')
        # Shared eval input is label-free, including no category/teacher lookup path.
        (args.output_dir/f'eval-{cohort}.json').write_text(json.dumps({'rows':[
            {k:r[k] for k in ['id','split','features']} for r in entries]},indent=2)+'\n')
    print(json.dumps({'constant_prefix_length':canonical['prefix_length'],'classes':len(design['active_classes'])}),flush=True)


if __name__=='__main__':main()
