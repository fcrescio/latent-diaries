import pytest

torch = pytest.importorskip('torch')

from latent_diaries.image_query_reader import ImageQueryReader, ridge_length
from latent_diaries.sequence_reader import masked_prefix_loss


def test_query_gradients_future_isolation_and_length_independence():
    torch.manual_seed(42)
    coef = torch.zeros(769)
    coef[0] = 3
    reader = ImageQueryReader(6, torch.zeros(1024), torch.ones(1024), coef, width=16).eval()
    features = torch.randn(2, 768)
    prefix, mask, _ = reader(features)
    masked_prefix_loss(prefix, torch.randn_like(prefix), mask, reader.scale).backward()
    assert reader.memory.weight.grad.norm() > 0
    assert reader.queries.grad.norm() > 0
    assert reader.decoder.layers[-1].multihead_attn.in_proj_weight.grad.norm() > 0
    with torch.no_grad():
        before = reader(features)[0]
        reader.length_coef[0] = 5
        after, _, lengths = reader(features)
        assert torch.equal(before, after) and lengths.tolist() == [5, 5]
        reader.queries[:, 3:] += torch.randn_like(reader.queries[:, 3:])
        assert torch.allclose(after[:, :3], reader(features)[0][:, :3], atol=1e-6)
        assert torch.allclose(reader(features[:1])[0], reader(features)[0][:1], atol=1e-6)


def test_ridge_analytical_loo_matches_explicit_refits():
    torch.manual_seed(17)
    x, y = torch.randn(8, 3), torch.randn(8) + 20
    coefficient, info = ridge_length(x, y, alphas=(.1, 1.))
    errors = []
    for i in range(8):
        keep = torch.arange(8) != i
        c, _ = ridge_length(x[keep], y[keep], alphas=(info['alpha'],))
        prediction = torch.cat([torch.ones(1), x[i]]) @ c
        errors.append((prediction-y[i]).square())
    assert abs(torch.stack(errors).mean().item() - info['training_loocv_mse']) < 1e-4
    assert coefficient.shape == (4,)
