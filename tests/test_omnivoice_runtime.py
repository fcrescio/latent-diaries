"""CPU-only real pinned APIs, tiny random weights, never labeled generated speech."""

from types import MethodType, SimpleNamespace

import pytest

omnivoice = pytest.importorskip("omnivoice")
import torch
from omnivoice.models.omnivoice import GenerationTask
from transformers import Qwen3Config

from latent_diaries.omnivoice_protocol import (
    audit_inputs,
    check_api,
    generation_fingerprint,
    raw_mask_inputs,
)


def test_pinned_api_and_raw_mask_boundary():
    check_api(omnivoice.OmniVoice, GenerationTask)
    model = SimpleNamespace(
        device=torch.device("cpu"), config=SimpleNamespace(num_audio_codebook=8, audio_mask_id=1024)
    )
    out = raw_mask_inputs(model, 100)
    audit = audit_inputs("mask-only", "", 100, None, None, None, None, out, 1024)
    assert out["input_ids"].shape == (1, 8, 100) and out["audio_mask"].shape == (1, 100)
    assert audit["non_audio_positions"] == 0
    out["input_ids"][0, 0, 0] = 1
    with pytest.raises(RuntimeError):
        audit_inputs("mask-only", "", 100, None, None, None, None, out, 1024)
    with pytest.raises(RuntimeError):
        audit_inputs("voice-empty", "caption", 100, "fixed", torch.zeros(8, 2), "it", None, out, 1024)


def test_tiny_actual_iterative_sampler_fills_all_codebooks_without_text():
    torch.set_num_threads(2)
    torch.manual_seed(1)
    config = omnivoice.OmniVoiceConfig(
        llm_config=Qwen3Config(
            # Upstream embeds all IDs before selecting audio/text positions.
            vocab_size=2048,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            max_position_embeddings=128,
        ),
        num_audio_codebook=8,
        audio_vocab_size=1025,
        audio_mask_id=1024,
    )
    model = omnivoice.OmniVoice(config).eval().requires_grad_(False)
    calls = []

    def prepare(
        self,
        text,
        num_target_tokens,
        ref_text=None,
        ref_audio_tokens=None,
        lang=None,
        instruct=None,
        denoise=True,
    ):
        out = raw_mask_inputs(self, num_target_tokens)
        calls.append(
            audit_inputs(
                "mask-only",
                text,
                num_target_tokens,
                ref_text,
                ref_audio_tokens,
                lang,
                instruct,
                out,
                1024,
            )
        )
        return out

    model._prepare_inference_inputs = MethodType(prepare, model)
    cfg = omnivoice.OmniVoiceGenerationConfig(
        num_step=32,
        guidance_scale=2.0,
        position_temperature=5.0,
        class_temperature=0.0,
        postprocess_output=False,
    )
    assert generation_fingerprint(cfg)["num_step"] == 32
    task = GenerationTask(
        batch_size=1,
        texts=[""],
        target_lens=[6],
        langs=[None],
        instructs=[None],
        ref_texts=[None],
        ref_audio_tokens=[None],
        ref_rms=[None],
    )
    with torch.inference_mode():
        codes = model._generate_iterative(task, cfg)[0]
    assert len(calls) == 1 and codes.shape == (8, 6) and codes.dtype == torch.long
    assert bool(((codes >= 0) & (codes < 1024)).all())
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
