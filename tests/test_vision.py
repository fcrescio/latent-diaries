import json

import pytest

torch = pytest.importorskip("torch")

from latent_diaries.vision import pool_global, read_image_manifest


def test_global_pooling_masks_normalization_and_degenerate_inputs():
    hidden = torch.zeros(2, 3, 768)
    hidden[0, 0, 0] = 1
    hidden[0, 1, 1] = 1
    hidden[0, 2] = float("nan")  # Padding must contribute nothing.
    hidden[1, :, 2] = 2
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    values = pool_global(hidden, mask)
    assert torch.allclose(values.norm(dim=-1), torch.ones(2))
    assert torch.allclose(values[0, :2], torch.full((2,), 2 ** -.5))
    assert values[1, 2] == 1
    with pytest.raises(ValueError, match="Empty"):
        pool_global(hidden, torch.zeros_like(mask))
    with pytest.raises(ValueError, match="zero"):
        pool_global(torch.zeros_like(hidden), mask)
    with pytest.raises(ValueError, match="768"):
        pool_global(hidden[:, :, :5], mask)


def test_image_manifest_rejects_transcript_and_duplicate_identity(tmp_path):
    path = tmp_path / "manifest.json"
    image = tmp_path / "image.png"
    image.write_bytes(b"fixture")
    row = {"id": "one", "image": str(image), "scene_group": "scene", "source": "authored", "license": "CC0"}
    path.write_text(json.dumps([row]))
    assert read_image_manifest(path) == [row]
    for rows in [[row | {"text": "a caption"}], [row, row]]:
        path.write_text(json.dumps(rows))
        with pytest.raises(ValueError):
            read_image_manifest(path)
