# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib==3.10.7"]
# ///
"""Render existing private JSON diagnostics; no model loading or training."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit-dir', type=Path, required=True)
    p.add_argument('--policy-dir', default='policy-analysis')
    args = p.parse_args();root = args.audit_dir
    traces = json.loads((root / 'ladder/prior-traces.json').read_text())
    by_level = defaultdict(list)
    for trace in traces:
        gold = [r for r in trace['gold_history'] if 'selected_normalized_mse' in r]
        free = [r for r in trace['free_running'] if 'selected_normalized_mse' in r]
        interventions = {key: mean(r['selected_normalized_mse'] for r in rows if 'selected_normalized_mse' in r)
                         for key, rows in trace['single_oracle_vector_resets'].items()}
        by_level[trace['level']].append({'image_id': trace['image_id'], 'seed': trace['seed'],
            'gold_selected_mse': mean(r['selected_normalized_mse'] for r in gold),
            'gold_best_mse': mean(r['oracle_best_normalized_mse'] for r in gold),
            'free_selected_mse': mean(r['selected_normalized_mse'] for r in free),
            'gold_routing_best_agreement': mean(r['routing_matches_best'] for r in gold),
            'free_routing_best_agreement': mean(r['routing_matches_best'] for r in free),
            'best_single_reset_mean_mse': min(interventions.values()),
            'best_reset_position': min(interventions, key=interventions.get),
            'gold_mean_routing_entropy': mean(r['routing_entropy'] for r in gold),
            'free_mean_routing_entropy': mean(r['routing_entropy'] for r in free),
            'first_gold_selected_mse': gold[0]['selected_normalized_mse'],
            'first_free_stop_position': next((r['position'] for r in trace['free_running'] if r['position'] > 0 and r['stop_probability'] >= .5), None)})
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for level in ['L2', 'L3', 'L4']:
        selected = [t for t in traces if t['level'] == level]
        length = min(len(t['free_running'])-1 for t in selected)
        for name, style in [('gold_history', '--'), ('free_running', '-')]:
            values = [mean(t[name][i]['selected_normalized_mse'] for t in selected) for i in range(length)]
            axes[0].plot(range(length), values, style, label=f'{level} {name}')
        axes[1].plot(range(length), [mean(t['free_running'][i]['hidden_rms_from_gold_history'] for t in selected)
                                   for i in range(length)], label=level)
        axes[2].plot(range(length), [mean(t['free_running'][i]['routing_entropy'] for t in selected)
                                   for i in range(length)], label=level)
    for ax, title in zip(axes, ['Selected-vector error', 'Hidden-state drift', 'Free routing entropy'], strict=True):
        ax.set_title(title);ax.set_xlabel('Prefix position');ax.legend(fontsize=6);ax.grid(alpha=.2)
    fig.suptitle('V24 fixed-checkpoint diagnostics: two training images × three seeds; no quality scores')
    fig.tight_layout();fig.savefig(root / 'prior-drift.png', dpi=160);fig.savefig(root / 'prior-drift.svg');plt.close(fig)
    policies_path = root / args.policy_dir / 'policies.json'
    policy_summary = []
    if policies_path.exists():
        policies = json.loads(policies_path.read_text())
        for row in policies['results']:
            summary = {k: row[k] for k in ['image_id', 'history', 'reader_index', 'frames', 'actual_generation_replay_parity']}
            for kind in ['codec0_raw', 'codec0_effective', 'residual_raw', 'residual_effective']:
                data = row[kind]
                summary[kind] = {'mean_clamped_kl': mean(r['kl_clamped_1e30'] for r in data),
                    'first32_mean_clamped_kl': mean(r['kl_clamped_1e30'] for r in data[:32]),
                    'first32_top1_disagreement_rate': mean(r['top1_disagreement'] for r in data[:32]),
                    'first32_teacher_mass_outside_student_support': mean(r['teacher_mass_outside_student_support'] for r in data[:32]),
                    'top1_disagreement_rate': mean(r['top1_disagreement'] for r in data),
                    'teacher_mass_outside_student_support': mean(r['teacher_mass_outside_student_support'] for r in data),
                    'mean_support_jaccard': mean(r['support_jaccard'] for r in data),
                    'selected_probability_mean_absolute_difference': mean(abs(r['selected_probability_difference']) for r in data),
                    'teacher_selected_mean_probability': mean(r['teacher_selected_probability'] for r in data),
                    'student_selected_mean_probability': mean(r['student_selected_probability'] for r in data)}
                if kind.startswith('codec0'):
                    summary[kind]['eos_probability_mean_absolute_difference'] = mean(
                        abs(r['student_eos_probability']-r['teacher_eos_probability']) for r in data)
            shifted = row['student_shift_from_previous_version_raw']
            summary['student_version_shift_mean_kl'] = mean(r['kl_clamped_1e30'] for r in shifted) if shifted else None
            policy_summary.append(summary)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        current = ('current_final_reader_greedy' if any(r['history'] == 'current_final_reader_greedy'
                   for r in policy_summary) else 'current_final_reader')
        for history in ['gold', 'cached_iteration_0', 'cached_iteration_1', current]:
            values = []
            mass = []
            for index in range(3):
                subset = [r for r in policy_summary if r['history'] == history and r['reader_index'] == index]
                values.append(mean(r['codec0_raw']['first32_mean_clamped_kl'] for r in subset))
                mass.append(mean(r['codec0_effective']['first32_teacher_mass_outside_student_support'] for r in subset))
            axes[0].plot(range(3), values, marker='o', label=history)
            axes[1].plot(range(3), mass, marker='o', label=history)
        for ax, title in zip(axes, ['First32 raw codec0 teacher KL', 'First32 teacher mass outside student support'], strict=True):
            ax.set_title(title);ax.set_xticks([0, 1, 2], ['V22', 'Repair 1', 'Repair 2']);ax.legend(fontsize=7);ax.grid(alpha=.2)
        fig.tight_layout();fig.savefig(root / 'policy-drift.png', dpi=160);fig.savefig(root / 'policy-drift.svg');plt.close(fig)
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        current_rows = [r for r in policies['results'] if r['history'] == current]
        length = min(r['frames'] for r in current_rows)
        for index in range(3):
            subset = [r for r in current_rows if r['reader_index'] == index]
            curve = [mean(r['codec0_raw'][i]['kl_clamped_1e30'] for r in subset) for i in range(length)]
            disagreement = [mean(r['codec0_effective'][i]['top1_disagreement'] for r in subset) for i in range(length)]
            cumulative, total = [], 0
            for value in disagreement:
                total += value;cumulative.append(total)
            axes[0].plot(range(length), curve, label=f'Reader {index}')
            axes[1].plot(range(length), cumulative, label=f'Reader {index}')
            axes[2].plot(range(length), [mean(r['codec0_effective'][i]['student_eos_probability'] for r in subset)
                                        for i in range(length)], label=f'Reader {index}')
        reference = [r for r in current_rows if r['reader_index'] == 2]
        axes[2].plot(range(length), [mean(r['codec0_effective'][i]['teacher_eos_probability'] for r in reference)
                                    for i in range(length)], '--', label='Teacher')
        for ax, title in zip(axes, ['Raw KL on same current history', 'Cumulative top1 disagreement', 'EOS probability (not stopping success)'], strict=True):
            ax.set_title(title);ax.set_xlabel('Audio frame');ax.legend(fontsize=7);ax.grid(alpha=.2)
        fig.tight_layout();fig.savefig(root / 'policy-frames.png', dpi=160);fig.savefig(root / 'policy-frames.svg');plt.close(fig)
    representation = json.loads((root / 'ladder/representation.json').read_text())
    rows = representation['rows']
    representation_summary = {'training_images': len(rows), 'extended_alpha_selected': representation['extended_alpha_selected'],
        'extended_cv': representation['extended_cv'],
        'd1_d2_mean_cosine': mean(r['d1_d2_mean_cosine'] for r in rows),
        'd2_d3_mean_cosine': mean(r['d2_d3_cosine'] for r in rows),
        'd3_pre_normalization_mean_norm': mean(r['d3_pre_normalization_norm'] for r in rows),
        'd3_to_own_d1_retrieval_rate': mean(r['d3_to_d1']['own_retrieved'] for r in rows),
        'd3_to_own_d2_retrieval_rate': mean(r['d3_to_d2']['own_retrieved'] for r in rows)}
    outcome = {'scope': 'Numerical diagnostics only; no human quality, intelligibility or grounding scores',
        'plotting_matplotlib_version': matplotlib.__version__, 'prior': dict(by_level),
        'representation': representation_summary, 'policies': policy_summary}
    (root / 'diagnostic-summary.json').write_text(json.dumps(outcome, indent=2) + '\n')
    print(json.dumps({'traces': len(traces), 'policies': len(policy_summary), 'plots_written': True}))


if __name__ == '__main__':
    main()
