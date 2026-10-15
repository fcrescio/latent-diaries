import pytest

torch = pytest.importorskip('torch')

from latent_diaries.semantic_adapter import fit_coefficient, select_alpha


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
