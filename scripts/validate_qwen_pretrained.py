"""Local pretrained mechanical controls; no training or speech-quality claim.

Run in the worker after M0. Teacher text is used only to establish stock
embedding-assembly parity, never as a visual inference implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from qwen_tts import Qwen3TTSModel

from latent_diaries.conditioning import SoftConditioningBuilder, VisualCondition
from latent_diaries.qwen_baseline import DEFAULT_TEXT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    args = parser.parse_args()
    dtype = getattr(torch, args.dtype)
    rtol, atol = (1e-4, 1e-4) if dtype == torch.float32 else (0.02, 0.05)
    baseline = json.loads(Path("artifacts/m0_italian.json").read_text())
    snapshot = snapshot_download(
        baseline["model_id"], revision=baseline["model_revision"], local_files_only=True
    )
    wrapper = Qwen3TTSModel.from_pretrained(
        snapshot, device_map="cuda:0", dtype=dtype, attn_implementation="sdpa"
    )
    model = wrapper.model.eval().requires_grad_(False)
    talker = model.talker
    prompt = wrapper.create_voice_clone_prompt(
        ref_audio=os.environ["LD_REF_AUDIO"], x_vector_only_mode=True
    )
    ids = wrapper._tokenize_texts([wrapper._build_assistant_text(DEFAULT_TEXT)])[0]
    with torch.no_grad():
        soft = talker.text_projection(talker.get_text_embeddings()(ids[:, 3:-5]))
    soft = soft.float().detach().requires_grad_(True)
    condition = VisualCondition(
        soft,
        torch.ones(soft.shape[:2], dtype=torch.bool, device=soft.device),
        prompt[0].ref_spk_embedding[None].to(soft.device),
    )
    builder = SoftConditioningBuilder(model, tuple(ids[0, :3].tolist()))
    captured = {}

    class PrefillCaptured(Exception):
        pass

    original_generate = talker.generate

    def capture(**kwargs):
        captured.update(kwargs)
        raise PrefillCaptured()

    talker.generate = capture
    try:
        wrapper.generate_voice_clone(
            text=DEFAULT_TEXT,
            language="Italian",
            voice_clone_prompt=prompt,
            non_streaming_mode=True,
            max_new_tokens=4,
        )
    except PrefillCaptured:
        pass
    finally:
        talker.generate = original_generate
    assert captured, "Official prefill was not captured"
    text_calls = []
    handle = talker.get_text_embeddings().register_forward_pre_hook(
        lambda module, args: text_calls.append(args[0].detach().clone())
    )
    try:
        prefill = builder.build_prefill(condition)
        for key in ["inputs_embeds", "attention_mask", "trailing_text_hidden", "tts_pad_embed"]:
            torch.testing.assert_close(prefill[key], captured[key], rtol=0, atol=0)
        allowed = [
            ids[:, :3],
            torch.tensor(
                [
                    [
                        model.config.tts_bos_token_id,
                        model.config.tts_eos_token_id,
                        model.config.tts_pad_token_id,
                    ]
                ],
                device=ids.device,
            ),
        ]
        assert all(any(torch.equal(call, fixed) for fixed in allowed) for call in text_calls)
    finally:
        handle.remove()
    gold = model.speech_tokenizer.encode("artifacts/m0_italian.wav").audio_codes[0][:3]
    gold = gold[None].to(soft.device)
    batch = builder.build_teacher_forced(condition, gold)
    output = talker(**batch)
    output.loss.backward()
    grad_norm = float(soft.grad.float().norm())
    assert grad_norm > 0 and torch.isfinite(soft.grad).all()
    assert all(p.grad is None for p in model.parameters())
    with torch.no_grad():
        pref = talker(**prefill, use_cache=True, output_hidden_states=True)
        length = prefill["inputs_embeds"].shape[1]
        torch.testing.assert_close(
            pref.logits[:, -1], output.logits[:, length - 1], rtol=rtol, atol=atol
        )
        stepped = talker(
            input_ids=gold[:, 0, :1],
            past_key_values=pref.past_key_values,
            past_hidden=pref.past_hidden,
            generation_step=pref.generation_step,
            trailing_text_hidden=pref.trailing_text_hidden,
            tts_pad_embed=prefill["tts_pad_embed"],
            attention_mask=torch.ones(1, length + 1, dtype=torch.long, device=soft.device),
            cache_position=torch.tensor([length], device=soft.device),
            use_cache=True,
            subtalker_dosample=False,
            subtalker_top_p=1.0,
            subtalker_top_k=0,
            subtalker_temperature=1.0,
            output_hidden_states=True,
        )
        replay_codes = stepped.hidden_states[-1]
        assert replay_codes.shape == (1, 16)
        replay = talker(**builder.build_teacher_forced(condition, replay_codes[:, None]))
        delta = float((stepped.logits[:, -1].float() - replay.logits[:, length].float()).abs().max())
        torch.testing.assert_close(stepped.logits[:, -1], replay.logits[:, length], rtol=rtol, atol=atol)
    manifest = {
        "scope": "Pretrained Qwen mechanics; not visual speech or learned inversion",
        "model_id": baseline["model_id"],
        "model_revision": baseline["model_revision"],
        "device": str(soft.device),
        "dtype": str(prefill["inputs_embeds"].dtype),
        "hidden_size": soft.shape[-1],
        "prefix_length": length,
        "teacher_frames": gold.shape[1],
        "loss": float(output.loss),
        "soft_gradient_norm": grad_norm,
        "frozen_model_gradients_absent": True,
        "exact_stock_prefill_parity": True,
        "fixed_text_ids_only_in_builder": True,
        "cache_replay_max_abs_difference": delta,
        "cache_replay_tolerance": {"rtol": rtol, "atol": atol},
        "codebooks": replay_codes.shape[-1],
        "lock_sha256": hashlib.sha256(Path("uv.lock").read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    Path("artifacts/qwen-pretrained-validation.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
