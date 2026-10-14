import json
from pathlib import Path
import pytest
from latent_diaries import hf_ops


def test_no_public_hf_repo(monkeypatch):
    monkeypatch.setenv("HF_PRIVATE", "false")
    monkeypatch.setenv("HF_NAMESPACE", "someone")
    monkeypatch.setenv("HF_MODEL_REPO", "latent-diaries")
    with pytest.raises(RuntimeError, match="Public repos"):
        hf_ops.ensure_private_repo(None, "model")


def test_no_dataset_upload_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("LD_APPROVE_DATASET_UPLOAD", raising=False)
    with pytest.raises(RuntimeError, match="LD_APPROVE_DATASET_UPLOAD"):
        hf_ops.publish_dataset(None, tmp_path)


def test_capabilities_fail_closed():
    c=json.loads((Path(__file__).parents[1]/"docs"/"capabilities.json").read_text())
    assert c["human_review_service_implemented"]
    assert not c["image_reader_training_implemented"]
