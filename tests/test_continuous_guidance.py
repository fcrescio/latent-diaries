import pytest

torch = pytest.importorskip('torch')

from latent_diaries.multihypothesis_prior import MultihypothesisPrior, continuous_log_ratio


def test_density_ratio_ignores_internal_index_permutation():
    torch.manual_seed(7)
    means, null_means = torch.randn(5, 8), torch.randn(6, 8)
    routing, null_routing, scale = torch.randn(5), torch.randn(6), torch.ones(8)
    value = continuous_log_ratio(means, routing, null_means, null_routing, scale, 12)
    order = torch.tensor([3, 1, 5, 0, 2, 4])
    other = continuous_log_ratio(means, routing, null_means[order], null_routing[order], scale, 12)
    torch.testing.assert_close(value, other)
    neutral = continuous_log_ratio(means, routing, means, routing, scale, 12)
    assert torch.equal(neutral, torch.zeros_like(neutral))
    assert (value.abs() <= 2).all()


def test_zero_condition_guidance_is_neutral_and_reproducible():
    torch.manual_seed(42)
    basis = torch.linalg.qr(torch.randn(12, 4)).Q.T
    model = MultihypothesisPrior(torch.zeros(12), torch.ones(12), basis, torch.ones(4),
                                modes=5, hidden=16, maximum=4).eval()
    features = torch.zeros(1, 768)
    original, _ = model.sample(features, sample_modes=True)
    guided, _ = model.sample(features, sample_modes=True, guidance_scale=1, guidance_steps=3)
    assert torch.equal(original, guided)
    image = torch.randn(1, 768)
    first, _ = model.sample(image, sample_modes=True, guidance_scale=1, guidance_steps=3)
    second, _ = model.sample(image, sample_modes=True, guidance_scale=1, guidance_steps=3)
    assert torch.equal(first, second) and torch.isfinite(first).all()
