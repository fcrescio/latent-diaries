"""Reject invalid experiment provenance before loading any expensive model."""
import importlib.util
import json
from pathlib import Path

import pytest

torch = pytest.importorskip('torch')
from safetensors.torch import save_file

from latent_diaries.inversion import digest


def script(name, monkeypatch):
    directory = Path(__file__).parents[1]/'scripts'
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location(name, directory/(name+'.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'load_backbone', lambda *_: pytest.fail('Model loading must not occur'))
    return module


def test_rollout_collection_rejects_reserved_data(tmp_path, monkeypatch):
    import sys
    dataset = tmp_path/'dataset.json'
    dataset.write_text(json.dumps({'rows':[{'id':'reserved', 'split':'fresh'}]}))
    run = tmp_path/'reader';run.mkdir()
    (run/'reader.safetensors').write_bytes(b'checksum-only test')
    (run/'config.json').write_text(json.dumps({'dataset_sha256':digest(dataset),
        'reader_sha256':digest(run/'reader.safetensors'), 'talker_lora_rank':0}))
    module = script('cache_student_histories', monkeypatch)
    monkeypatch.setattr(sys, 'argv', ['cache', '--dataset',str(dataset),'--reader-run',str(run),
        '--design',str(tmp_path/'missing-design'),'--teachers',str(tmp_path/'missing-teachers'),
        '--output-dir',str(tmp_path/'output')])
    with pytest.raises(ValueError, match='reserved/fresh'):
        module.main()
    assert not (tmp_path/'output').exists()


def test_repair_rejects_rollouts_from_another_reader(tmp_path, monkeypatch):
    import sys
    dataset = tmp_path/'dataset.json'
    dataset.write_text(json.dumps({'rows':[{'id':'train','split':'train','scene_group':'g0'},
        {'id':'dev','split':'heldout','scene_group':'g1'}]}))
    run = tmp_path/'reader';run.mkdir();(run/'reader.safetensors').write_bytes(b'checksum-only test')
    lock = digest(Path('uv.lock'))
    info = {'model_id':'test', 'model_revision':'test', 'lock_sha256':lock,
        'evaluation_scope':'development', 'reader_sha256':digest(run/'reader.safetensors'),
        'dataset_sha256':digest(dataset)}
    (run/'config.json').write_text(json.dumps(info))
    anchor = tmp_path/'anchor';anchor.mkdir();(anchor/'config.json').write_text(json.dumps(info))
    save_file({'soft_tokens':torch.zeros(1,2,1024)},str(anchor/'prefix.safetensors'))
    teachers = tmp_path/'teachers';teachers.mkdir()
    (teachers/'manifest.json').write_text(json.dumps({'records':[], 'dataset_sha256':digest(dataset),'lock_sha256':lock}))
    policy = tmp_path/'policies';policy.mkdir()
    (policy/'manifest.json').write_text(json.dumps({'records':[], 'teacher_manifest_sha256':digest(teachers/'manifest.json'),'lock_sha256':lock}))
    rollout = tmp_path/'rollouts';rollout.mkdir()
    (rollout/'manifest.json').write_text(json.dumps({'dataset_sha256':'wrong-reader-data'}))
    module = script('train_policy_repair', monkeypatch)
    monkeypatch.setattr(sys, 'argv', ['train','--dataset',str(dataset),'--teachers',str(teachers),
        '--policy-cache',str(policy),'--rollout-cache',str(rollout),'--provenance-run',str(run),
        '--anchor-run',str(anchor),'--run-dir',str(tmp_path/'output'),'--steps','1'])
    with pytest.raises(ValueError, match='Rollout bank'):
        module.main()
