import pytest

torch = pytest.importorskip("torch")

from latent_diaries.image_reader import GlobalImageReader
from latent_diaries.sequence_reader import masked_prefix_loss


def test_image_reader_gradient_checkpoint_and_batch_parity(tmp_path):
    torch.manual_seed(42)
    coefficients = torch.zeros(769)
    coefficients[0] = 3
    model = GlobalImageReader(5, torch.zeros(1024), torch.ones(1024), coefficients, width=8)
    features = torch.randn(2, 768)
    output, mask, lengths = model(features)
    assert output.shape == (2, 5, 1024)
    assert lengths.tolist() == [3, 3]
    assert not mask[:, 3:].any()
    loss = masked_prefix_loss(output, torch.randn_like(output), mask, model.scale)
    loss.backward()
    assert model.network[1].weight.grad.norm() > 0
    assert model.network[-1].weight.grad.reshape(5, 1024, 8)[3:].count_nonzero() == 0
    torch.save(model.state_dict(), tmp_path / "reader.pt")
    copy = GlobalImageReader(5, torch.zeros(1024), torch.ones(1024), coefficients, width=8)
    copy.load_state_dict(torch.load(tmp_path / "reader.pt", weights_only=True))
    with torch.no_grad():
        assert torch.equal(model(features)[0], copy(features)[0])
        assert torch.allclose(model(features[:1])[0], model(features)[0][:1], atol=1e-6)
    with pytest.raises(ValueError, match="768"):
        model(features[:, :5])
