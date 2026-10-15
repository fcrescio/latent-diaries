import pytest

torch = pytest.importorskip('torch')

from latent_diaries.multihypothesis_prior import MultihypothesisPrior
from latent_diaries.prior_calibration import NumericImageAdapter, opening_nll, shared_opening_length


@pytest.mark.parametrize('device', ['cpu'] + (['cuda'] if torch.cuda.is_available() else []))
def test_frozen_prior_transmits_adapter_gradient_without_future_leak(device):
    torch.manual_seed(42)
    basis = torch.linalg.qr(torch.randn(12, 4)).Q.T
    prior = MultihypothesisPrior(torch.zeros(12), torch.ones(12), basis, torch.ones(4),
                                modes=3, hidden=16, maximum=4).to(device).eval().requires_grad_(False)
    adapter = NumericImageAdapter(torch.randn(769, 768) * 0.01).to(device)
    image = torch.randn(2, 768, device=device)
    target = torch.randn(2, 3, 12, device=device)
    lengths = torch.tensor([3, 2], device=device)
    loss = opening_nll(prior, adapter(image), target, lengths).mean()
    loss.backward()
    assert adapter.coefficient.grad.norm() > 0
    assert torch.isfinite(adapter.coefficient.grad).all()
    assert all(p.grad is None for p in prior.parameters())
    one = torch.ones(2, dtype=torch.long, device=device)
    changed = target.clone();changed[:, 1:] += 100
    assert torch.equal(opening_nll(prior, adapter(image), target, one),
                       opening_nll(prior, adapter(image), changed, one))


def test_opening_boundary_uses_common_vectors_before_continuations():
    values = torch.zeros(3, 8, 12)
    values[:, 4:] = torch.arange(3)[:, None, None]
    assert shared_opening_length(values, torch.tensor([7, 6, 8])) == 4
    with pytest.raises(ValueError):
        shared_opening_length(values[:1], torch.tensor([7]))
