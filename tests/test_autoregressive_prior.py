import pytest

torch = pytest.importorskip('torch')
from latent_diaries.autoregressive_prior import AutoregressivePrior, causal_loss


def test_causal_prefix_context_and_frozen_numeric_basis():
    torch.manual_seed(15)
    model = AutoregressivePrior(torch.zeros(8), torch.ones(8), torch.eye(8), torch.ones(8),
                                hidden=16, width=16, maximum=4)
    features = torch.randn(2, 768)
    previous = torch.randn(2, 5, 8)
    before = model.contexts(features, previous)
    changed = previous.clone();changed[:, 3:] += 2
    torch.testing.assert_close(before[:, :3], model.contexts(features, changed)[:, :3])
    target = torch.randn(2, 4, 8)
    loss, metrics = causal_loss(model, features, target, torch.tensor([2, 4]))
    assert torch.isfinite(loss) and float(metrics['stop_bce'].detach()) > 0
    loss.backward()
    assert model.gru.weight_ih_l0.grad.abs().sum() > 0
    assert model.flow_state.weight.grad.abs().sum() > 0
    assert model.stop.weight.grad.abs().sum() > 0
    assert model.basis.grad is None
    first, _ = model.sample(features[:1], seed=17, steps=3)
    second, _ = model.sample(features[:1], seed=17, steps=3)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert 1 <= first.shape[1] <= 4
