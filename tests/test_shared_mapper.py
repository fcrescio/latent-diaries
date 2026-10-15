import pytest

torch = pytest.importorskip("torch")

from latent_diaries.prefix_regression import predict_prefix
from latent_diaries.shared_mapper import SharedMapper, validate_split


def test_shared_mapper_gradients_and_unseen_numeric_input():
    base = torch.randn(1, 4, 8, requires_grad=True)
    mapper = SharedMapper(base, width=12)
    inputs = torch.tensor([[-1.0, 1.0, -1.0], [1.0, -1.0, 1.0]])
    output = mapper(inputs)
    assert output.shape == (2, 4, 8)
    assert not torch.equal(output[0], output[1])
    output.square().mean().backward()
    assert base.grad is None
    assert mapper.net[0].weight.grad.abs().sum() > 0
    assert mapper.net[-1].weight.grad.abs().sum() > 0
    assert mapper(torch.zeros(1, 3)).isfinite().all()
    with pytest.raises(ValueError):
        mapper(torch.tensor([[float("nan"), 0.0, 0.0]]))


def test_split_rejects_feature_leakage_and_duplicate_ids():
    rows = [
        {"id": "a", "split": "train", "features": [-1, 1, 1]},
        {"id": "b", "split": "heldout", "features": [1, 1, 1]},
    ]
    assert len(validate_split(rows)[0]) == 1
    with pytest.raises(ValueError, match="Duplicate features"):
        validate_split([rows[0], rows[1] | {"features": rows[0]["features"]}])
    with pytest.raises(ValueError, match="unique"):
        validate_split([rows[0], rows[1] | {"id": "a"}])


def test_additive_mapper_composes_directions_without_interactions():
    mapper = SharedMapper(torch.randn(1, 4, 8), width=0)
    x = torch.tensor([[1.0, -1.0, 0.0]])
    y = torch.tensor([[0.0, 0.0, 1.0]])
    torch.testing.assert_close(mapper(x + y), mapper(x) + mapper(y) - mapper(torch.zeros_like(x)))
    assert sum(p.numel() for p in mapper.parameters()) == 4 * 8 * 4
    mapper(x).square().mean().backward()
    assert mapper.net[0].weight.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="Width"):
        SharedMapper(torch.zeros(1, 4, 8), width=-1)


def test_numeric_prefix_length_preserves_only_predicted_positions():
    coef = torch.arange(24, dtype=torch.float32).reshape(4, 3, 2)
    length_coef = torch.tensor([2.0, 1.0, 0.0, 0.0])
    prefix, length = predict_prefix(torch.tensor([-1.0, 0.0, 0.0]), coef, length_coef)
    assert length == 1 and prefix.shape == (1, 1, 2)
    torch.testing.assert_close(prefix[0], (coef[0] - coef[1])[:1])
    assert predict_prefix(torch.tensor([1.0, 0.0, 0.0]), coef, length_coef)[1] == 3
    assert predict_prefix(torch.tensor([-9.0, 0.0, 0.0]), coef, length_coef)[1] == 1
    with pytest.raises(ValueError, match="Finite"):
        predict_prefix(torch.tensor([float("nan"), 0.0, 0.0]), coef, length_coef)
