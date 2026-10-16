import pytest

torch = pytest.importorskip('torch')
from latent_diaries.reader_rank import reconstruction_rank_bound


def test_known_rank_floor_and_variable_mask_denominator():
    # Four centered points on orthogonal axes have two equal squared singular values2.
    targets = [torch.tensor([p]) for p in [[1., 0.], [-1., 0.], [0., 1.], [0., -1.]]]
    scale = torch.ones(2)
    assert reconstruction_rank_bound(targets, scale, 1)['full_masked_normalized_mse_lower_bound'] == pytest.approx(.25)
    assert reconstruction_rank_bound(targets, scale, 2)['full_masked_normalized_mse_lower_bound'] == pytest.approx(0, abs=1e-12)
    targets[0] = torch.cat([targets[0], torch.tensor([[123., 456.]])])
    # Extra positions do not enter the common block, but do enter the full masked denominator.
    assert reconstruction_rank_bound(targets, scale, 1)['full_masked_normalized_mse_lower_bound'] == pytest.approx(.2)
