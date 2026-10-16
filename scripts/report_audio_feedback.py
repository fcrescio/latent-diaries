# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib==3.10.7"]
# ///
"""Export whitelisted aggregate statistics and plots; never private media or labels."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    paths = {name: args.root/path for name, path in {
        'primary': 'evaluation-uncertainty/report.json', 'confirmation': 'confirmation/report.json',
        'translation': 'translation-probe/report.json', 'order': 'memory-order/report.json',
        'detail': 'detail-report.json', 'selection': 'rerank-selection/results.json',
        'smoke_fp32': 'smoke-fp32/manifest.json', 'smoke_bf16': 'smoke-bf16/manifest.json',
        'asr_tiny': 'lexical-audit.json', 'asr_small': 'lexical-audit-small.json'}.items()}
    data = {k: json.loads(path.read_text()) for k, path in paths.items()}
    d = data['primary'];translation = data['translation'];detail = data['detail']
    selections = data['selection']['results']
    null_margins = [max(r['candidate_image_audio_scores'])-max(r['null_generation_image_audio_scores'])
                    for r in selections]
    summary = {
        'scope': 'Frozen offline semantic feasibility and an unapproved no-training selection pilot',
        'report_source_sha256': digest(Path(__file__)),
        'private_evidence_sha256': {k: digest(path) for k, path in paths.items()},
        'phase0': {name: {k: data[name][k] for k in [
            'model_id', 'model_revision', 'dtype', 'parameter_count', 'all_parameters_frozen',
            'peak_allocated_vram_bytes', 'metrics', 'torch_version', 'transformers_version',
            'vision_lock_sha256', 'worker_lock_sha256', 'variable_text_used']}
            for name in ['smoke_fp32', 'smoke_bf16']},
        'pilot_cases': d['training_cases'], 'pilot_capture_groups': d['independent_capture_groups'],
        'audio_text_top1': d['audio_to_text']['Document']['top1'],
        'attribute_pairs': {k: v for k, v in d['attribute_pairs']['Document'].items()
                            if k not in {'similarities', 'per_source_balanced_accuracy'}},
        'negation_pairs': {k: v for k, v in d['negation_pairs']['Document'].items()
                           if k not in {'similarities', 'per_source_balanced_accuracy'}},
        'image_audio_top1': d['image_to_audio']['top1'],
        'within_scene_image_audio_top1': d['within_historical_scene_image_audio'],
        'visual_attribute_preference': detail['visual_preference_original_over_attribute_change'],
        'isolated_batch_max_abs_error': detail['isolated_batch_audio_embeddings_max_abs_error'],
        'isolated_single_max_abs_error': detail['isolated_single_audio_embeddings_max_abs_error'],
        'perturbation_audio_text_top1': {k: v['audio_to_text']['top1'] for k, v in d['perturbations'].items()},
        'rated_gibberish_preference': d['human_rated_degraded_controls']['degraded_preference'],
        'joint_shared_image_diagnostics': d['joint_shared_image_dominance'],
        'confirmation_cases': data['confirmation']['cases'],
        'confirmation_unique_strings': data['confirmation']['unique_reference_strings'],
        'confirmation': data['confirmation']['results'],
        'translated_reference_audio_text_top1': translation['audio_to_translated_text']['top1'],
        'translated_attribute_pair_accuracy': translation['translated_attribute_pairs']['balanced_audio_to_text_accuracy'],
        'translation_human_validated': translation['translation_human_validated'],
        'auxiliary_lexical_audits': {name: {'model': data[name]['asr_model_id'],
            'aggregate': data[name]['aggregate'], 'changed_selection_or_gates': False,
            'human_pronunciation_validated': False} for name in ['asr_tiny', 'asr_small']},
        'order_diagnostics': data['order']['metrics'], 'numeric_gates': d['gates'],
        'reranking': {'development_images': 8, 'candidates_per_image': 4,
                      'additional_semantic_null_candidates': 4, 'blinded_pairs': 28,
                      'max_choice_differs_from_first': sum(r['methods']['image_audio_max']['candidate_index'] !=
                          r['methods']['first_candidate']['candidate_index'] for r in selections),
                      'max_choice_changes_with_shuffled_scorer': sum(r['methods']['image_audio_max']['candidate_index'] !=
                          r['methods']['shuffled_image_scorer']['candidate_index'] for r in selections),
                      'image_conditioned_max_beats_null_pool_max': sum(m > 0 for m in null_margins),
                      'mean_image_conditioned_max_minus_null_max': sum(null_margins)/len(null_margins),
                      'human_quality_improvement_verified': False},
        'limits': ['Training teachers are not unseen image generalization.',
                   'Historical teacher captions are automatic annotations, not per-image human truth.',
                   'Additional confirmation cases can share capture groups with the pilot.',
                   'Fixed-catalog cluster bootstrap does not establish unseen-data error bounds.',
                   'Rated gibberish controls contain only two source images.',
                   'Reference timbre perturbations are not independent speakers.',
                   'Translations are assistant-authored diagnostic references.',
                   'Cumulative audio embeddings do not preserve ordered narrative state.',
                   'Candidate cosine gain is a selection objective, not a human quality result.'],
        'temporal_reader_training_started': False, 'reserved_fresh_access': False,
        'private_media_uploaded': False, 'm2_accepted': False}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout='constrained')
    values = [('Audio / text', summary['audio_text_top1']),
              ('Changed attribute', summary['attribute_pairs']['balanced_audio_to_text_accuracy']),
              ('Image / audio', summary['image_audio_top1']),
              ('Same-scene image / audio', summary['within_scene_image_audio_top1']),
              ('Visual attribute preference', summary['visual_attribute_preference'])]
    def bars(ax, values):
        means = [m['mean'] for _, m in values]
        errors = [[m['mean']-m['ci95'][0] for _, m in values],
                  [m['ci95'][1]-m['mean'] for _, m in values]]
        ax.barh(range(len(values)), means, xerr=errors, color='#368785', capsize=3)
        ax.set_yticks(range(len(values)), [k for k, _ in values]);ax.invert_yaxis()
        ax.set_xlim(0, 1.04);ax.grid(axis='x', alpha=.2)
    bars(axes[0], values);axes[0].set_title('30 training cases: distinct questions')
    robustness = summary['perturbation_audio_text_top1']
    bars(axes[1], [(k, robustness[k]) for k in ['clean', 'gain_half', 'noise20', 'speed09',
                                               'speed11', 'reverse', 'shuffle100ms', 'phase_scramble', 'crop1s', 'last2s']])
    axes[1].set_title('Audio / text retrieval under perturbation')
    order = summary['order_diagnostics']
    bars(axes[2], [('AB vs BA cosine', order['AB_BA_cosine']),
                  ('Last-fragment preference', order['last_segment_preference_balanced']),
                  ('AA vs A cosine', order['AA_single_cosine'])])
    axes[2].set_title('Pooled audio loses temporal structure')
    fig.suptitle('Frozen semantic feedback feasibility — not a narrative quality evaluation')
    fig.savefig(args.output_dir/'feasibility.png', dpi=160)
    plt.close(fig)
    print('Wrote aggregate-only summary and plot; no private captions, IDs, media or tensors exported')


if __name__ == '__main__':
    main()
