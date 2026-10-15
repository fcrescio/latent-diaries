import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from latent_diaries.sequence_reader import SequenceReader, masked_prefix_loss
from latent_diaries.shared_mapper import validate_split


def test_sequence_reader_gradients_masks_and_checkpoint(tmp_path):
    torch.manual_seed(42)
    reader = SequenceReader(
        3,
        6,
        torch.zeros(8),
        torch.ones(8),
        torch.tensor([4.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
        width=16,
        layers=2,
        heads=2,
    )
    features = torch.tensor([[1.0, 0.0, 0.0, -1.0, 1.0], [0.0, 0.0, 1.0, 1.0, -1.0]])
    prefix, mask, lengths = reader(features)
    assert lengths.tolist() == [3, 5]
    assert mask.sum(1).tolist() == [3, 5]
    prefix.retain_grad()
    target = torch.randn_like(prefix)
    masked_prefix_loss(prefix, target, mask, reader.scale).backward()
    assert prefix.grad[~mask].abs().sum() == 0
    assert reader.weather.weight.grad.abs().sum() > 0
    assert reader.queries.grad.abs().sum() > 0
    assert reader.decoder.layers[-1].multihead_attn.in_proj_weight.grad.abs().sum() > 0
    reader.eval()
    with torch.no_grad():
        expected = reader(features)[0]
        for i in range(len(features)):
            torch.testing.assert_close(
                reader(features[i : i + 1])[0][0], expected[i], rtol=1e-5, atol=1e-6
            )
    torch.save(reader.state_dict(), tmp_path / "state.pt")
    restored = SequenceReader(
        3, 6, torch.zeros(8), torch.ones(8), torch.zeros(6), width=16, layers=2, heads=2
    )
    restored.load_state_dict(torch.load(tmp_path / "state.pt", weights_only=True))
    restored.eval()
    with torch.no_grad():
        torch.testing.assert_close(restored(features)[0], expected, rtol=0, atol=0)
    # Invalid queries cannot influence valid ones through masked self-attention.
    with torch.no_grad():
        reader.queries[:, 5] += 100
        torch.testing.assert_close(reader(features)[0][mask], expected[mask], rtol=1e-5, atol=1e-6)


def test_sequence_v3_has_fresh_tests_and_broader_factor_coverage():
    design = json.loads(Path("configs/research/compositions-v3.json").read_text())
    train, test = validate_split(design["rows"])
    assert len(train) == 22 and len(test) == 2
    assert {r["id"] for r in test} == {"w02", "e01"}
    for col in range(8):
        assert {r["features"][col] for r in test} <= {r["features"][col] for r in train}
    for weather in [4, 5]:
        assert sum(r["features"][weather] == 1 for r in train) == 3
