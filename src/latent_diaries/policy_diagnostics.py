"""Compare policies on identical action histories, including real decoding filters."""


def policy_comparison(student, teacher, selected, eos=None):
    if student.shape != teacher.shape or student.ndim != 2:
        raise ValueError('Aligned [positions,vocabulary] policies required')
    p, q = teacher.float().softmax(-1), student.float().softmax(-1)
    # Exact KL may be infinite after top-k support changes. Report lost mass too.
    absent = q == 0
    lost_mass = p.masked_fill(~absent, 0).sum(-1)
    finite_kl = (p * (p.clamp_min(1e-30).log() - q.clamp_min(1e-30).log())).sum(-1)
    rows = []
    for i in range(len(p)):
        p_support, q_support = p[i] > 0, q[i] > 0
        union = (p_support | q_support).sum()
        row = {'position': i, 'kl_clamped_1e30': float(finite_kl[i]),
               'exact_kl_infinite': bool(lost_mass[i] > 0), 'teacher_mass_outside_student_support': float(lost_mass[i]),
               'support_jaccard': float((p_support & q_support).sum() / union),
               'top1_disagreement': int(p[i].argmax()) != int(q[i].argmax()),
               'teacher_selected_probability': float(p[i, selected[i]]),
               'student_selected_probability': float(q[i, selected[i]])}
        row['selected_probability_difference'] = row['student_selected_probability'] - row['teacher_selected_probability']
        if eos is not None:
            row.update(teacher_eos_probability=float(p[i, eos]), student_eos_probability=float(q[i, eos]))
        rows.append(row)
    return rows
