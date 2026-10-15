import pytest

torch = pytest.importorskip('torch')
from latent_diaries.multihypothesis_prior import MultihypothesisPrior, multihypothesis_loss


def test_continuous_candidates_and_routing_gradients():
    torch.manual_seed(20)
    model = MultihypothesisPrior(torch.zeros(8), torch.ones(8), torch.eye(8), torch.ones(8),
                                modes=3, hidden=16, maximum=4)
    features = torch.randn(2, 768)
    targets = torch.randn(2, 4, 8)
    loss, terms = multihypothesis_loss(model, features, targets, torch.tensor([2, 4]))
    loss.backward()
    assert model.candidates[-1].weight.grad.abs().sum() > 0
    assert model.routing.weight.grad.abs().sum() > 0
    assert model.gru.weight_ih_l0.grad.abs().sum() > 0
    assert all(torch.isfinite(v) for v in terms.values())
    first, _ = model.sample(features[:1])
    second, _ = model.sample(features[:1])
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert 1 <= first.shape[1] <= 4
