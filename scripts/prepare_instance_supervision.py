"""Cache image-specific training prefixes; reuse existing image-only dev vectors."""
import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from latent_diaries.inversion import digest, load_backbone


def select_training_labels(rows, labels):
    """Reject held-out/duplicate supervision before loading any model."""
    by_id = {r['id']: r for r in rows}
    selected = []
    seen = set()
    for record in labels:
        key = record['id']
        if key in seen or key not in by_id or by_id[key]['split'] != 'train':
            raise ValueError('Unique existing training IDs required')
        seen.add(key)
        if record.get('training_only') is not True:
            raise ValueError('Training-only provenance required')
        label = record['label']
        if type(label.get('keep')) is not bool:
            raise ValueError('Boolean keep required')
        if label['keep']:
            text = label.get('narration_it')
            if not isinstance(text, str) or not 8 <= len(text.split()) <= 18:
                raise ValueError('Expected one short training narration')
            selected.append((by_id[key], text))
    if len(selected) < 4:
        raise ValueError('At least four accepted training examples required')
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if not Path('/.dockerenv').exists() or args.output_dir.exists():
        raise ValueError('Use managed worker and a new output directory')
    original = json.loads(args.dataset.read_text())
    if any(r['split'] not in {'train', 'heldout'} for r in original['rows']):
        raise ValueError('Fresh access forbidden')
    selected = select_training_labels(original['rows'], [json.loads(line) for line in args.labels.read_text().splitlines()])
    info = json.loads(Path('artifacts/m0_italian.json').read_text())
    wrapper, builder, _ = load_backbone(info)
    args.output_dir.mkdir(parents=True)
    rows = []
    for row, text in selected:
        folder = args.output_dir / row['id']
        folder.mkdir()
        ids = wrapper._tokenize_texts([wrapper._build_assistant_text(text)])[0]
        with torch.no_grad():
            prefix = wrapper.model.talker.text_projection(wrapper.model.talker.get_text_embeddings()(ids[:, 3:-5]))
        save_file({'soft_tokens': prefix.cpu()}, str(folder / 'prefix.safetensors'))
        metadata = {k: info[k] for k in ['model_id', 'model_revision']}
        metadata.update(lock_sha256=digest(Path('uv.lock')), role_ids=list(builder.role_ids))
        (folder / 'config.json').write_text(json.dumps(metadata, indent=2)+'\n')
        rows.append({**row, 'cache_dir': str(folder)})
    # Development vectors carry no teacher text. Fresh reserved candidates are absent.
    rows += [r for r in original['rows'] if r['split'] == 'heldout']
    metadata = {k: original[k] for k in ['design_sha256', 'image_features_sha256', 'vision_manifest_sha256']}
    metadata.update(rows=rows, labels_sha256=digest(args.labels), parent_dataset_sha256=digest(args.dataset),
                    source_sha256=digest(Path(__file__)), scope='Image-specific development supervision; no generalization acceptance')
    (args.output_dir / 'dataset.json').write_text(json.dumps(metadata, indent=2)+'\n')
    (args.output_dir / 'eval-inputs.json').write_text(json.dumps({'rows': [
        {k: r[k] for k in ['id', 'split', 'features']} for r in rows]}, indent=2)+'\n')
    print(json.dumps({'training_examples': len(selected), 'development_vectors': len(rows)-len(selected)}), flush=True)


if __name__ == '__main__':
    main()
