import pytest

torch = pytest.importorskip('torch')
from latent_diaries.visual_diagnostic import classification_metrics, ridge_scores


def test_dual_ridge_matches_centered_primal_and_preserves_query_batch():
    torch.manual_seed(11)
    x, query = torch.randn(9, 4), torch.randn(3, 4)
    labels = torch.tensor([0, 1, 2] * 3)
    y = torch.nn.functional.one_hot(labels, 3).double()
    centered = x.double() - x.double().mean(0)
    weights = torch.linalg.solve(centered.T @ centered + torch.eye(4), centered.T @ (y - y.mean(0)))
    expected = (query.double() - x.double().mean(0)) @ weights + y.mean(0)
    got = ridge_scores(x, labels, query, 3)
    torch.testing.assert_close(got, expected)
    torch.testing.assert_close(got[:1], ridge_scores(x, labels, query[:1], 3))
    metric = classification_metrics(torch.eye(3), torch.arange(3))
    assert metric['macro_recall'] == metric['accuracy'] == 1
