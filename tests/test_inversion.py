"""Optimizer/checkpoint continuation must preserve the next update exactly."""

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("qwen_tts")
from safetensors.torch import load_file

from latent_diaries.inversion import digest, save_checkpoint


def test_checkpoint_preserves_adam_continuation(tmp_path):
    torch.manual_seed(17)
    prefix = torch.nn.Parameter(torch.randn(1, 2, 3))
    optimizer = torch.optim.AdamW([prefix], lr=0.02)

    def step(parameter, opt):
        opt.zero_grad(set_to_none=True)
        parameter.square().sum().backward()
        opt.step()

    step(prefix, optimizer)
    save_checkpoint(tmp_path, prefix, optimizer, 1, {"seed": 17})
    state = torch.load(tmp_path / "state.pt", weights_only=True)
    assert state["prefix_sha256"] == digest(tmp_path / "prefix.safetensors")
    assert json.loads((tmp_path / "config.json").read_text())["completed_steps"] == 1
    restored = torch.nn.Parameter(load_file(str(tmp_path / "prefix.safetensors"))["soft_tokens"])
    restored_optimizer = torch.optim.AdamW([restored], lr=0.02)
    restored_optimizer.load_state_dict(state["optimizer"])
    step(prefix, optimizer)
    step(restored, restored_optimizer)
    torch.testing.assert_close(restored, prefix, rtol=0, atol=0)
