import pytest

torch = pytest.importorskip('torch')
from latent_diaries.multihypothesis_prior import MultihypothesisPrior
from latent_diaries.prior_diagnostics import compare_traces, trace_prior


def test_trace_matches_real_sampler_until_stop_and_intervention_is_causal():
    torch.manual_seed(4)
    model = MultihypothesisPrior(torch.zeros(8), torch.ones(8), torch.eye(8), torch.ones(8), modes=4, maximum=5).eval()
    semantic = torch.randn(1, 768)
    gold = torch.randn(1, 5, 8)
    free = trace_prior(model, semantic, gold, 42)
    sampled, _ = model.sample(semantic, seed=42, sample_modes=True)
    torch.testing.assert_close(sampled, free['prefix'][:, :sampled.shape[1]], rtol=0, atol=0)
    reset = trace_prior(model, semantic, gold, 42, reset_position=2)
    torch.testing.assert_close(free['prefix'][:, :3], reset['prefix'][:, :3], rtol=0, atol=0)
    assert not torch.equal(free['contexts'][3], reset['contexts'][3])
    reference = trace_prior(model, semantic, gold, 42, gold_history=True)
    metrics = compare_traces(reference, free)
    assert metrics[0]['hidden_rms_from_gold_history'] == 0
    assert metrics[1]['hidden_rms_from_gold_history'] > 0
