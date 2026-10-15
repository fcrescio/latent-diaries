"""Freeze authored text-prior combinations and project targets in managed worker."""
import argparse
import json
import random
from pathlib import Path

import torch
from safetensors.torch import save_file

from latent_diaries.inversion import digest, load_backbone
from latent_diaries.prior_corpus import candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if not Path('/.dockerenv').exists() or args.output_dir.exists():
        raise ValueError('Use managed worker and a new corpus directory')
    info = json.loads(Path('artifacts/m0_italian.json').read_text())
    wrapper, builder, _ = load_backbone(info)
    grouped = {}
    for row in candidates():
        ids = wrapper._tokenize_texts([wrapper._build_assistant_text(row['text'])])[0][:, 3:-5]
        if ids.shape[1] <= 32:
            grouped.setdefault((row['category'], row['split']), []).append(row | {'ids': ids})
    selected = []
    for (category, split), rows in sorted(grouped.items()):
        random.Random(42).shuffle(rows)
        limit = 128 if split == 'train' else 16
        if len(rows) < limit:
            raise ValueError('Insufficient authored combinations under token cap')
        selected.extend(rows[:limit])
    if len({r['text'] for r in selected}) != len(selected):
        raise ValueError('Duplicate utterances')
    prefixes, rows = [], []
    pad_id = torch.tensor([[wrapper.model.config.tts_pad_token_id]], device=wrapper.model.device)
    with torch.inference_mode():
        pad = wrapper.model.talker.text_projection(wrapper.model.talker.get_text_embeddings()(pad_id))
        for i, row in enumerate(selected):
            ids = row.pop('ids')
            prefix = wrapper.model.talker.text_projection(wrapper.model.talker.get_text_embeddings()(ids))
            length = prefix.shape[1]
            value = pad.expand(1, 32, -1).clone()
            value[:, :length] = prefix
            prefixes.append(value[0].cpu())
            rows.append(row | {'id': f's{i:04d}', 'length': length})
    args.output_dir.mkdir(parents=True)
    save_file({'prefixes': torch.stack(prefixes)}, str(args.output_dir / 'prefixes.safetensors'))
    provenance = {k: info[k] for k in ['model_id', 'model_revision']}
    provenance.update(lock_sha256=digest(Path('uv.lock')), role_ids=list(builder.role_ids),
                      source_sha256=digest(Path(__file__)), corpus_source_sha256=digest(Path('src/latent_diaries/prior_corpus.py')),
                      prefix_sha256=digest(args.output_dir / 'prefixes.safetensors'))
    (args.output_dir / 'corpus.json').write_text(json.dumps({'rows': rows, 'provenance': provenance},
                                                         ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'train': sum(r['split'] == 'train' for r in rows),
                      'heldout': sum(r['split'] == 'heldout' for r in rows),
                      'minimum_length': min(r['length'] for r in rows),
                      'maximum_length': max(r['length'] for r in rows)}), flush=True)


if __name__ == '__main__':
    main()
