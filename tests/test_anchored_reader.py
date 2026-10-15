import pytest

torch = pytest.importorskip('torch')
from latent_diaries.anchored_reader import initialization_fingerprint, make_anchored_reader


def test_fixed_numeric_anchor_and_image_gradient():
    torch.manual_seed(42)
    anchor = torch.randn(5, 1024)
    reader = make_anchored_reader(anchor, width=16)
    features = torch.randn(2, 768)
    before, mask, lengths = reader(features)
    torch.testing.assert_close(before, anchor[None].expand(2, -1, -1), atol=1e-6, rtol=1e-6)
    assert mask.all() and lengths.tolist() == [5, 5]
    checksum = initialization_fingerprint(reader)
    # Opposing audio-domain gradients can open the initially zero image residual.
    loss = (before[0] * torch.randn_like(anchor)).mean() - (before[1] * torch.randn_like(anchor)).mean()
    loss.backward()
    assert reader.network[-1].weight.grad.abs().sum() > 0
    torch.optim.SGD(reader.parameters(), lr=.1).step()
    after = reader(features)[0]
    assert not torch.allclose(after[0], after[1])
    assert initialization_fingerprint(reader) != checksum
