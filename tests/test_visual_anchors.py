import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

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


@pytest.mark.parametrize('mode', ['opening-v10', 'opening-v11'])
def test_labeler_rejects_nontraining_before_starting_service(tmp_path, mode):
    inventory = tmp_path / 'inventory.json'
    inventory.write_text(json.dumps([{'id': 'held', 'split': 'heldout', 'path': 'unused'}]))
    result = subprocess.run([sys.executable, 'scripts/label_private_images.py', '--mode', mode,
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


def test_short_choice_prompt_and_scene_membership():
    prompt = anchor_prompt('food')
    assert all(f'food:{i}' in prompt for i in range(6))
    assert 'vehicles:0' not in prompt
    assert validate_anchor_label({'keep': True, 'anchor': 'food:0'}, 'food')['keep']
    with pytest.raises(ValueError):
        validate_anchor_label({'keep': True, 'anchor': 'food:0'}, 'vehicles')
    with pytest.raises(ValueError):
        anchor_prompt('unknown')


@pytest.mark.parametrize('response', ['not json', '{"keep":true,"anchor":"food:999"}',
                                     '{"keep":true,"anchor":"vehicles:0"}'])
def test_invalid_model_label_is_excluded_and_service_stopped(tmp_path, monkeypatch, response):
    spec = importlib.util.spec_from_file_location('local_labeler', Path('scripts/label_private_images.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    image = tmp_path / 'sample.jpg'
    Image.new('RGB', (4, 4)).save(image)
    inventory = tmp_path / 'inventory.json'
    inventory.write_text(json.dumps([{'id': 'train1', 'split': 'train', 'path': str(image), 'scene_hint': 'food'}]))
    output = tmp_path / 'labels.jsonl'
    actions = []
    monkeypatch.setattr(module.subprocess, 'run', lambda cmd, **kw: actions.append(cmd[1]))
    monkeypatch.setattr(module.signal, 'signal', lambda *args: None)
    monkeypatch.setattr(module, 'json_request', lambda route, payload=None:
                        {'data': [{'id': 'mock-local'}]} if route == '/v1/models' else
                        {'choices': [{'message': {'content': response}}]})
    monkeypatch.setattr(sys, 'argv', ['labeler', '--inventory', str(inventory), '--output', str(output),
                                    '--mode', 'opening-v11', '--docker', 'fake-docker'])
    module.main()
    row = json.loads(output.read_text())
    assert row['label'] == {'keep': False, 'anchor': None}
    assert row['validation_rejection'] == 'invalid_model_anchor_response'
    assert len(row['labeler_source_sha256']) == 64
    assert actions == ['start', 'stop']
