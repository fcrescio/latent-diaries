"""Read-only POST-REVIEW prefix diagnosis; all per-image results remain private.

Held-out texts may be projected here only after the experiment is reviewed and
frozen. They never update reader weights. Text-derived controls are not learned
image-conditioned successes.
"""
import argparse
import json
from pathlib import Path

import torch
from image_probe import make_reader
from safetensors.torch import load_file

from latent_diaries.inversion import digest, load_backbone, render


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--design', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if not Path('/.dockerenv').exists():
        raise RuntimeError('Use managed Docker worker')
    if args.output_dir.exists():
        raise ValueError('Use fresh post-review diagnostic directory')
    info = json.loads((args.run_dir / 'config.json').read_text())
    weights = args.run_dir / 'reader.safetensors'
    before = digest(weights)
    assert before == info['reader_sha256']
    reader = make_reader(info, load_file(str(weights))).eval().requires_grad_(False)
    design = {r['id']: r for r in json.loads(args.design.read_text())['rows']}
    rows = json.loads(args.dataset.read_text())['rows']
    plan = json.loads(args.plan.read_text())
    wrapper, builder, speaker = load_backbone(info)
    args.output_dir.mkdir(parents=True)
    measured, cached = [], {}
    with torch.inference_mode():
        for row in rows:
            ids = wrapper._tokenize_texts([wrapper._build_assistant_text(design[row['id']]['teacher_text'])])[0]
            target = wrapper.model.talker.text_projection(
                wrapper.model.talker.get_text_embeddings()(ids[:, 3:-5]))
            prediction, _, lengths = reader(torch.tensor([row['features']], dtype=torch.float32))
            target = target.cpu()
            n = min(target.shape[1], int(lengths[0]))
            errors = (prediction[0, :n] - target[0, :n]).square().mean(-1)
            # Holdout lengths were not available during fitting; only diagnostic now.
            measured.append({'id': row['id'], 'split': row['split'],
                             'predicted_length': int(lengths[0]), 'teacher_length': target.shape[1],
                             'length_matches': int(lengths[0]) == target.shape[1],
                             'aligned_prefix_mse': float(errors.mean()),
                             'first_five_mse': float(errors[:5].mean()),
                             'remaining_mse': float(errors[5:].mean()) if n > 5 else None})
            cached[row['id']] = (prediction, target)
    summary = {}
    for split in ['train', 'heldout']:
        selected = [r for r in measured if r['split'] == split]
        summary[split] = {'count': len(selected), 'length_matches': sum(r['length_matches'] for r in selected),
                          'aligned_prefix_mse': sum(r['aligned_prefix_mse'] for r in selected)/len(selected)}
    controls = []
    chosen = [plan['training_id'], plan['heldout_ids'][0]]
    for i, label in enumerate(chosen):
        _, target = cached[label]
        output = render(builder, wrapper, target.to(speaker.device), speaker,
                        args.output_dir / f'text_positive_{i}.wav', 105, 42)
        controls.append({'id': label, 'kind': 'exact_text_prefix_positive_control',
                         'learned_image_output': False, **output})
    # Correcting length alone isolates truncation from vector reconstruction error.
    label = chosen[1]
    predicted, target = cached[label]
    if target.shape[1] <= predicted.shape[1]:
        output = render(builder, wrapper, predicted[:, :target.shape[1]].to(speaker.device), speaker,
                        args.output_dir / 'oracle_length.wav', 105, 42)
        controls.append({'id': label, 'kind': 'predicted_vectors_oracle_text_length',
                         'learned_image_success_claim': False, **output})
    assert digest(weights) == before
    result = {'post_review_only': True, 'weights_updated': False, 'weights_sha256': before,
              'source_sha256': digest(Path(__file__)), 'summary': summary,
              'results': measured, 'controls': controls,
              'scope': 'Tensor/control diagnosis, not human quality or word alignment'}
    (args.output_dir / 'diagnostic.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
