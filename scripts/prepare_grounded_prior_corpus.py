"""Mix image-specific training narratives with the verified Italian replay corpus."""
import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from latent_diaries.grounded_corpus import valid_narrative
from latent_diaries.inversion import digest, load_backbone


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--narratives', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--replay-corpus', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if not Path('/.dockerenv').exists() or args.output_dir.exists():
        raise ValueError('Managed worker and fresh corpus required')
    image_rows = json.loads(args.dataset.read_text())['rows']
    training_ids = {r['id'] for r in image_rows if r['split']=='train'}
    annotations = [json.loads(line) for line in args.narratives.read_text().splitlines() if line]
    seen = set()
    for row in annotations:
        if row['id'] in seen or row['id'] not in training_ids or row.get('training_only') is not True:
            raise ValueError('Unique existing training-image narrative groups required')
        seen.add(row['id'])
    replay_file = args.replay_corpus/'corpus.json'
    replay = json.loads(replay_file.read_text())
    replay_prefix = args.replay_corpus/'prefixes.safetensors'
    if digest(replay_prefix) != replay['provenance']['prefix_sha256']:
        raise ValueError('Replay targets changed')
    info = json.loads(Path('artifacts/m0_italian.json').read_text())
    for key in ['model_id','model_revision']:
        if info[key] != replay['provenance'][key]:raise ValueError('Replay backbone differs')
    wrapper,builder,_ = load_backbone(info)
    if list(builder.role_ids) != replay['provenance']['role_ids']:
        raise ValueError('Replay role protocol differs')
    prefix_values = list(load_file(str(replay_prefix))['prefixes'].unbind(0))
    rows = [{**row,'id':'replay_'+row['id']} for row in replay['rows']]
    used = {r['text'] for r in rows}
    groups = []
    with torch.inference_mode():
        ids = torch.tensor([[wrapper.model.config.tts_pad_token_id]],device=wrapper.model.device)
        pad = wrapper.model.talker.text_projection(wrapper.model.talker.get_text_embeddings()(ids))
        for annotation in annotations:
            accepted = []
            for text in annotation['variants']:
                if not valid_narrative(text) or text in used or any(text==r[0] for r in accepted):continue
                ids = wrapper._tokenize_texts([wrapper._build_assistant_text(text)])[0][:,3:-5]
                if not 1 <= ids.shape[1] <= 32:continue
                prefix = wrapper.model.talker.text_projection(wrapper.model.talker.get_text_embeddings()(ids))
                value = pad.expand(1,32,-1).clone();value[:,:prefix.shape[1]]=prefix
                accepted.append((text,value[0].cpu(),prefix.shape[1]))
            if len(accepted) < 5:continue
            group = 'image_'+annotation['id'];groups.append(annotation['id'])
            held = 2 if len(accepted)>=8 else 1
            for i,(text,prefix,length) in enumerate(accepted):
                rows.append({'id':f'{group}_v{i}', 'category':group,'opening_index':0,'ending_index':i,
                    'split':'heldout' if i>=len(accepted)-held else 'train','text':text,'length':length})
                prefix_values.append(prefix);used.add(text)
    if len(groups)<100:raise ValueError('Insufficient validated image-specific narrative coverage')
    args.output_dir.mkdir(parents=True)
    save_file({'prefixes':torch.stack(prefix_values).contiguous()},str(args.output_dir/'prefixes.safetensors'))
    provenance = {k:info[k] for k in ['model_id','model_revision']}
    provenance.update(lock_sha256=digest(Path('uv.lock')),role_ids=list(builder.role_ids),
        prefix_sha256=digest(args.output_dir/'prefixes.safetensors'),source_sha256=digest(Path(__file__)),
        parent_corpus_sha256=digest(replay_file),narratives_sha256=digest(args.narratives),
        image_dataset_sha256=digest(args.dataset),image_groups=groups,
        scope='Training-image narratives plus Italian replay; heldout rows are language paraphrases, not heldout-image labels')
    (args.output_dir/'corpus.json').write_text(json.dumps({'rows':rows,'provenance':provenance},ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'image_groups':len(groups),'utterances':len(rows),'train':sum(r['split']=='train' for r in rows),'language_heldout':sum(r['split']=='heldout' for r in rows)}),flush=True)


if __name__=='__main__':main()
