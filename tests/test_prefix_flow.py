import pytest

torch = pytest.importorskip('torch')
from latent_diaries.prefix_flow import PrefixFlow, flow_loss


def test_continuous_flow_gradients_numeric_sampling_and_length():
    torch.manual_seed(3)
    model = PrefixFlow(torch.zeros(8), torch.ones(8), torch.zeros(32),
                       torch.eye(32), torch.ones(32), width=48)
    features = torch.randn(2, 768)
    prefix = torch.randn(2, 4, 8)
    lengths = torch.tensor([2, 4])
    clean = model.pack(prefix, lengths)
    assert clean[:, -1].tolist() == [0, 1]
    recovered, recovered_lengths = model.unpack(clean)
    torch.testing.assert_close(recovered, prefix)
    assert torch.equal(recovered_lengths, lengths)
    noise = torch.randn_like(clean)
    velocity = model.velocity(noise * 0.6 + clean * 0.4, features, torch.full((2,), 0.4))
    loss = flow_loss(velocity, clean - noise)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    first, lens = model.sample(features, seed=12, steps=3)
    second, lens2 = model.sample(features, seed=12, steps=3)
    torch.testing.assert_close(first, second, atol=0, rtol=0)
    assert torch.equal(lens, lens2) and bool(((lens >= 1) & (lens <= 4)).all())
    altered = features.clone()
    altered[:, 0] += 0.5
    changed = model.sample(altered, seed=12, steps=3)[0]
    assert not torch.equal(first, changed)
    with pytest.raises(ValueError):
        model.velocity(noise, features[:, :12], torch.ones(2))
