import pytest

torch = pytest.importorskip('torch')
from latent_diaries.policy_diagnostics import policy_comparison


def test_truncated_support_does_not_hide_infinite_kl_or_selected_action_loss():
    teacher = torch.tensor([[2., 1., -float('inf')]])
    student = torch.tensor([[-float('inf'), 1., 2.]])
    row = policy_comparison(student, teacher, [0], eos=2)[0]
    assert row['exact_kl_infinite'] and row['teacher_mass_outside_student_support'] > 0.7
    assert row['student_selected_probability'] == 0
    assert row['support_jaccard'] == pytest.approx(1/3)
    same = policy_comparison(teacher, teacher, [0])[0]
    assert same['kl_clamped_1e30'] == 0 and not same['exact_kl_infinite']
