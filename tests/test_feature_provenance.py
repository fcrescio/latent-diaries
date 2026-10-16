import json

import pytest

torch = pytest.importorskip('torch')
from safetensors.torch import save_file

from latent_diaries.feature_provenance import checksum, verify_image_features, verify_text_features


def test_text_cannot_be_relabeled_as_image_by_cli_mode(tmp_path):
    vector = torch.nn.functional.normalize(torch.ones(1, 768), dim=-1)
    weights = tmp_path / 'features.safetensors'
    save_file({'embeddings': vector}, str(weights))
    manifest = {'model_id': 'google/embeddinggemma-2', 'model_revision': 'a' * 40,
                'variable_text_used': False, 'records': [{'id': 'a'}],
                'features_sha256': checksum(weights)}
    path = tmp_path / 'manifest.json';path.write_text(json.dumps(manifest))
    rows = [{'id': 'a', 'features': vector[0].tolist()}]
    assert verify_image_features(rows, [tmp_path])
    rows[0]['features'][0] += 0.01
    with pytest.raises(ValueError, match='differs'):
        verify_image_features(rows, [tmp_path])
    manifest['variable_text_used'] = True;path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='image-only'):
        verify_image_features(rows, [tmp_path])
    manifest = {'ids': ['a'], 'corpus_sha256': 'corpus', 'features_sha256': checksum(weights)}
    path.write_text(json.dumps(manifest));rows[0]['features'] = vector[0].tolist()
    assert verify_text_features(rows, tmp_path, 'corpus')
    with pytest.raises(ValueError):
        verify_image_features(rows, [tmp_path])
