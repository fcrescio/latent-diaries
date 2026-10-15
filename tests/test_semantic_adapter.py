import pytest

torch = pytest.importorskip('torch')

from latent_diaries.semantic_adapter import fit_coefficient, scale_semantic_deviation, select_alpha


def test_group_validation_and_generalization():
    generator = torch.Generator().manual_seed(9)
    centers = torch.randn(18, 5, generator=generator)
    x = centers.repeat_interleave(2, 0)
    y = torch.nn.functional.normalize(x @ torch.randn(5, 7, generator=generator), dim=-1)
    groups = [str(i) for i in range(18) for _ in range(2)]
    alpha, scores = select_alpha(x, y, groups)
    assert len(scores) == 4 and all(torch.isfinite(torch.tensor(v)) for v in scores.values())
    assert (alpha, scores) == select_alpha(x, y, groups)
    weights = fit_coefficient(x, y, alpha)
    assert weights.shape == (6, 7)
    prediction = torch.nn.functional.normalize(torch.cat([torch.ones(len(x), 1), x], -1) @ weights, dim=-1)
    assert float((prediction * y).sum(-1).mean()) > 0.9
    with pytest.raises(ValueError):
        select_alpha(x, y, ['same'] * len(x))


def test_gain_preserves_training_center_and_scales_only_deviation():
    torch.manual_seed(3)
    x, y = torch.randn(12, 4), torch.randn(12, 7)
    coefficient = fit_coefficient(x, y, 1)
    mean = y.double().mean(0).float()
    assert torch.equal(coefficient, scale_semantic_deviation(coefficient, mean, 1))
    larger = scale_semantic_deviation(coefficient, mean, 3)
    query = torch.cat([torch.ones(len(x), 1), x], -1)
    torch.testing.assert_close(query @ larger - mean, 3 * (query @ coefficient - mean))
    center = torch.cat([torch.ones(1), x.mean(0)])
    torch.testing.assert_close(center @ larger, center @ coefficient)
