"""Verify numeric inputs against frozen encoder caches, independently of mode flags."""
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_image_features(rows, folders):
    indexed, evidence = {}, []
    for folder in map(Path, folders):
        manifest_path, weights = folder / 'manifest.json', folder / 'features.safetensors'
        manifest = json.loads(manifest_path.read_text())
        if (manifest.get('model_id') != 'google/embeddinggemma-2'
                or manifest.get('variable_text_used') is not False
                or checksum(weights) != manifest['features_sha256']):
            raise ValueError('Verified image-only encoder cache required')
        vectors = load_file(str(weights))['embeddings']
        records = manifest['records']
        if vectors.shape != (len(records), 768):
            raise ValueError('Image cache dimensions differ')
        for record, vector in zip(records, vectors, strict=True):
            if record['id'] in indexed:
                raise ValueError('Duplicate cache image identity')
            indexed[record['id']] = vector
        evidence.append({'manifest_sha256': checksum(manifest_path),
                         'features_sha256': checksum(weights), 'model_revision': manifest['model_revision']})
    for row in rows:
        value = torch.tensor(row['features'], dtype=torch.float32)
        if row['id'] not in indexed or not torch.equal(value, indexed[row['id']].float()):
            raise ValueError('Input vector differs from verified image encoder cache')
    return evidence


def verify_text_features(rows, folder, corpus_sha256):
    folder = Path(folder)
    manifest = json.loads((folder / 'manifest.json').read_text())
    weights = folder / 'features.safetensors'
    if (manifest['corpus_sha256'] != corpus_sha256
            or checksum(weights) != manifest['features_sha256']):
        raise ValueError('Training text cache provenance differs')
    vectors = load_file(str(weights))['embeddings']
    indexed = dict(zip(manifest['ids'], vectors, strict=True))
    for row in rows:
        if row['id'] not in indexed or not torch.equal(torch.tensor(row['features']), indexed[row['id']]):
            raise ValueError('Input differs from verified training text cache')
    return {'manifest_sha256': checksum(folder / 'manifest.json'), 'features_sha256': checksum(weights)}
