"""Training provenance boundaries for image-specific speech supervision."""
import importlib.util
from pathlib import Path

import pytest

# Script has ML imports: exercised in the single managed worker.
pytest.importorskip('torch')
spec = importlib.util.spec_from_file_location('instance_supervision', Path(__file__).parents[1] / 'scripts/prepare_instance_supervision.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture():
    rows = [{'id': str(i), 'split': 'train'} for i in range(4)]
    labels = [{'id': str(i), 'training_only': True, 'label': {
        'keep': True, 'narration_it': 'Il pane dorato riposa sul tavolo accanto alla finestra.'}} for i in range(4)]
    return rows, labels


def test_instance_training_boundary():
    rows, labels = fixture()
    assert len(module.select_training_labels(rows, labels)) == 4
    rows[0]['split'] = 'heldout'
    with pytest.raises(ValueError, match='training IDs'):
        module.select_training_labels(rows, labels)


def test_instance_duplicate_and_provenance():
    rows, labels = fixture()
    with pytest.raises(ValueError, match='training IDs'):
        module.select_training_labels(rows, labels + labels[:1])
    labels[0]['training_only'] = False
    with pytest.raises(ValueError, match='provenance'):
        module.select_training_labels(rows, labels)
