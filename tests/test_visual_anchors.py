import json
import subprocess
import sys

import pytest

from latent_diaries.visual_anchors import (
    anchor_choices,
    anchor_prompt,
    training_anchor_targets,
    validate_anchor_label,
)


def test_fixed_training_choices_and_schema():
    choices = anchor_choices()
    assert len(choices) == 60
    assert len(set(choices.values())) == 60
    assert all(key in anchor_prompt() for key in choices)
    for key in choices:
        assert validate_anchor_label({'keep': True, 'anchor': key})['anchor'] == key
    assert validate_anchor_label({'keep': False, 'anchor': None})['anchor'] is None
    for invalid in [{'keep': True, 'anchor': 'arbitrary text'}, {'keep': False, 'anchor': 'food:0'},
                    {'keep': 1, 'anchor': 'food:0'}, {'keep': True, 'anchor': ['food:0']},
                    {'keep': True, 'anchor': 'food:0', 'command': 'ignored'}]:
        with pytest.raises(ValueError):
            validate_anchor_label(invalid)


def test_labeler_rejects_nontraining_before_starting_service(tmp_path):
    inventory = tmp_path / 'inventory.json'
    inventory.write_text(json.dumps([{'id': 'held', 'split': 'heldout', 'path': 'unused'}]))
    result = subprocess.run([sys.executable, 'scripts/label_private_images.py', '--mode', 'opening-v10',
                             '--inventory', str(inventory), '--output', str(tmp_path / 'labels.jsonl'),
                             '--docker', '/must-not-be-called'], check=False, capture_output=True, text=True)
    assert result.returncode != 0
    assert 'training images only' in result.stderr
    assert not (tmp_path / 'labels.jsonl').exists()


def test_adapter_supervision_cannot_include_development_or_duplicates():
    good = {'id': 'train1', 'training_only': True, 'label': {'keep': True, 'anchor': 'food:0'}}
    assert training_anchor_targets([good], {'train1'}) == {'train1': 'food:0'}
    for rows in [[good, good], [good | {'id': 'held1'}], [good | {'training_only': False}]]:
        with pytest.raises(ValueError):
            training_anchor_targets(rows, {'train1'})
