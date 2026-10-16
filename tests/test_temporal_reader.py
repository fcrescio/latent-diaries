import pytest

torch = pytest.importorskip("torch")
from latent_diaries.temporal_reader import TemporalReader, prefix_objective


def test_history_mask_and_image_only_are_exact_invariances():
    torch.manual_seed(7)
    image, a, b = [torch.randn(2, 768) for _ in range(3)]
    reader = TemporalReader(4, torch.zeros(1024), torch.ones(1024), width=8)
    absent = torch.zeros(2, dtype=torch.bool)
    present = ~absent
    assert torch.equal(reader(image, a, absent)[0], reader(image, b, absent)[0])
    assert not torch.equal(reader(image, a, present)[0], reader(image, b, present)[0])
    assert not torch.equal(reader(image, a, present)[0], reader(image.flip(0), a, present)[0])
    reader.feedback = False
    assert torch.equal(reader(image, a, present)[0], reader(image, b, present)[0])
    assert torch.equal(reader(image, a, present)[0], reader(image, b, absent)[0])


def test_masked_supervision_gradients_and_roundtrip():
    torch.manual_seed(8)
    reader = TemporalReader(4, torch.zeros(1024), torch.ones(1024), width=8)
    image = torch.randn(2, 768, requires_grad=True)
    audio = torch.randn(2, 768, requires_grad=True)
    prefix, _, _, length = reader(image, audio, torch.ones(2, dtype=torch.bool))
    gold = torch.randn_like(prefix)
    lengths = torch.tensor([2, 3])
    mask = torch.arange(4)[None] < lengths[:, None]
    loss = prefix_objective(prefix, gold, mask, reader.scale, length, lengths)[0]
    changed = gold.clone()
    changed[~mask] = 100000
    assert torch.equal(loss, prefix_objective(prefix, changed, mask, reader.scale, length, lengths)[0])
    loss.backward()
    assert image.grad.abs().sum() > 0 and audio.grad.abs().sum() > 0
    assert reader.output.weight.grad.abs().sum() > 0 and reader.length.weight.grad.abs().sum() > 0
    clone = TemporalReader(4, reader.mean, reader.scale, width=8)
    clone.load_state_dict(reader.state_dict())
    assert torch.equal(prefix, clone(image, audio, torch.ones(2, dtype=torch.bool))[0])
    with pytest.raises(ValueError):
        reader(image, audio, torch.ones(2))
