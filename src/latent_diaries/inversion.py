"""M1 research control: one utterance, a learned continuous prefix, frozen Qwen.

This is memorization, not an image reader or generalization evidence. Training
and inference do not consume LD_REF_TEXT. Only fixed protocol IDs and a fixed
reference speaker vector are used alongside the learned continuous tokens.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from qwen_tts import Qwen3TTSModel
from safetensors.torch import load_file, save_file

from .conditioning import SoftConditioningBuilder, VisualCondition


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_backbone(metadata: dict):
    if not Path("/.dockerenv").exists():
        raise RuntimeError("Use the managed Docker worker for experiments")
    reference = os.environ.get("LD_REF_AUDIO", "")
    if not reference or not Path(reference).is_file():
        raise RuntimeError("A local user-provided reference voice is required")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    snapshot = snapshot_download(
        metadata["model_id"], revision=metadata["model_revision"], local_files_only=True
    )
    wrapper = Qwen3TTSModel.from_pretrained(
        snapshot, device_map=device, dtype=torch.float32, attn_implementation="sdpa"
    )
    wrapper.model.eval().requires_grad_(False)
    prompt = wrapper.create_voice_clone_prompt(ref_audio=reference, x_vector_only_mode=True)
    # This literal is a fixed protocol marker, never utterance/caption text.
    role = wrapper._tokenize_texts(["<|im_start|>assistant\n"])[0]
    role_ids = tuple(role[0].tolist())
    if len(role_ids) != 3:
        raise RuntimeError("Pinned tokenizer assistant-role protocol changed")
    speaker = prompt[0].ref_spk_embedding[None].to(device)
    return wrapper, SoftConditioningBuilder(wrapper.model, role_ids), speaker


def condition(prefix, speaker):
    return VisualCondition(
        prefix, torch.ones(prefix.shape[:2], device=prefix.device, dtype=torch.bool), speaker
    )


def aligned_residual_logits(talker, frame_hidden, gold):
    """Predict codebook j from hidden and gold codebooks <j, without label shift."""
    if gold.ndim != 2 or frame_hidden.ndim != 2 or gold.shape[0] != frame_hidden.shape[0]:
        raise ValueError("Expected frame-aligned [frames, codebooks] and [frames, hidden]")
    if gold.shape[1] != talker.config.num_code_groups:
        raise ValueError("Gold codebook count differs from pinned Talker config")
    inputs = [frame_hidden[:, None], talker.get_input_embeddings()(gold[:, :1])]
    embeddings = talker.code_predictor.get_input_embeddings()
    inputs.extend(embeddings[j - 1](gold[:, j : j + 1]) for j in range(1, gold.shape[1] - 1))
    return talker.code_predictor.forward_finetune(inputs_embeds=torch.cat(inputs, dim=1)).logits


@torch.no_grad()
def render(builder, wrapper, prefix, speaker, out: Path, max_frames: int, seed: int):
    import soundfile as sf

    codes = builder.generate_codes(condition(prefix, speaker), max_frames=max_frames, seed=seed)[0]
    if not len(codes):
        return {"frames": 0, "audio": None, "empty": True}
    wavs, sr = wrapper.model.speech_tokenizer.decode([{"audio_codes": codes}])
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out, wavs[0], sr)
    return {
        "frames": len(codes),
        "audio": str(out),
        "sample_rate": sr,
        "seconds": len(wavs[0]) / sr,
        "audio_sha256": digest(out),
        "codebook_count": codes.shape[1],
    }


def save_checkpoint(folder, prefix, optimizer, step, metadata):
    save_file({"soft_tokens": prefix.detach().cpu()}, str(folder / "prefix.tmp.safetensors"))
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "step": step,
            "prefix_sha256": digest(folder / "prefix.tmp.safetensors"),
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        },
        folder / "state.tmp.pt",
    )
    (folder / "prefix.tmp.safetensors").replace(folder / "prefix.safetensors")
    (folder / "state.tmp.pt").replace(folder / "state.pt")
    metadata["completed_steps"] = step
    (folder / "config.tmp.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (folder / "config.tmp.json").replace(folder / "config.json")


def train(
    target: Path,
    folder: Path,
    steps: int,
    tokens: int,
    learning_rate: float,
    seed: int,
    resume: bool,
    residual_loss_weight: float = 0.0,
):
    if steps < 1 or tokens < 1 or learning_rate <= 0 or residual_loss_weight < 0:
        raise ValueError("steps, tokens and learning rate must be positive")
    baseline = json.loads(Path("artifacts/m0_italian.json").read_text())
    metadata = {
        "stage": "M1.1 single-utterance inversion control",
        "model_id": baseline["model_id"],
        "model_revision": baseline["model_revision"],
        "seed": seed,
        "tokens": tokens,
        "learning_rate": learning_rate,
        "precision": "float32",
        "residual_loss_weight": residual_loss_weight,
        "target_sha256": digest(target),
        "lock_sha256": digest(Path("uv.lock")),
        "source_revision": os.environ.get("LD_CODE_REVISION"),
        "source_sha256": digest(Path(__file__)),
        "trainable_parameters": tokens * 1024,
        "human_listening_status": "pending",
        "image_conditioned": False,
    }
    if not resume and folder.exists() and any(folder.iterdir()):
        raise ValueError("Use a fresh run directory or explicit --resume")
    folder.mkdir(parents=True, exist_ok=True)
    if resume:
        saved = json.loads((folder / "config.json").read_text())
        for key in [
            "model_id",
            "model_revision",
            "seed",
            "tokens",
            "learning_rate",
            "precision",
            "residual_loss_weight",
            "target_sha256",
            "lock_sha256",
            "source_sha256",
        ]:
            if metadata[key] != saved[key]:
                raise ValueError(f"Resume configuration changed: {key}")
        metadata = saved
    torch.manual_seed(seed)
    wrapper, builder, speaker = load_backbone(metadata)
    device = speaker.device
    with torch.no_grad():
        gold = wrapper.model.speech_tokenizer.encode(str(target)).audio_codes[0][None].to(device)
        pad_id = torch.tensor([[wrapper.model.config.tts_pad_token_id]], device=device)
        pad = wrapper.model.talker.text_projection(wrapper.model.talker.get_text_embeddings()(pad_id))
        initial = pad.expand(1, tokens, -1).clone() + 0.02 * torch.randn(
            1, tokens, pad.shape[-1], device=device
        )
    torch.save(gold.cpu(), folder / "gold_codes.pt")
    prefix = torch.nn.Parameter(initial.clone())
    optimizer = torch.optim.AdamW([prefix], lr=learning_rate, weight_decay=0.0)
    start = 0
    if resume:
        prefix.data.copy_(load_file(str(folder / "prefix.safetensors"))["soft_tokens"].to(device))
        state = torch.load(folder / "state.pt", map_location=device, weights_only=True)
        if state["prefix_sha256"] != digest(folder / "prefix.safetensors"):
            raise RuntimeError("Checkpoint prefix/state pair is inconsistent")
        optimizer.load_state_dict(state["optimizer"])
        start = state["step"]
        if start != metadata["completed_steps"]:
            raise RuntimeError("Checkpoint pair is inconsistent; restore a complete checkpoint")
        torch.set_rng_state(state["cpu_rng"].cpu())
        if state["cuda_rng"]:
            torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda_rng"]])
    metadata["trainable_parameters"] = prefix.numel()
    metadata["target_frames"] = gold.shape[1]
    metadata["role_ids"] = list(builder.role_ids)
    started = time.perf_counter()
    with (folder / "loss.jsonl").open("a") as log:
        for step in range(start, steps):
            optimizer.zero_grad(set_to_none=True)
            batch = builder.build_teacher_forced(condition(prefix, speaker), gold)
            output = wrapper.model.talker(**batch, output_hidden_states=residual_loss_weight > 0)
            loss = output.loss
            residual_loss = torch.zeros((), device=device)
            if residual_loss_weight:
                prefix_length = builder.build_prefill(condition(prefix, speaker))["inputs_embeds"].shape[
                    1
                ]
                hidden = output.hidden_states[0][-1][
                    0, prefix_length - 1 : prefix_length - 1 + gold.shape[1]
                ]
                residual_logits = aligned_residual_logits(wrapper.model.talker, hidden, gold[0])
                residual_loss = torch.nn.functional.cross_entropy(
                    residual_logits.reshape(-1, residual_logits.shape[-1]), gold[0, :, 1:].reshape(-1)
                )
                loss = loss + residual_loss_weight * residual_loss
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite loss: inspect the run before retrying")
            loss.backward()
            if prefix.grad is None or not torch.isfinite(prefix.grad).all():
                raise RuntimeError("Missing/nonfinite continuous-token gradients")
            grad_norm = float(torch.nn.utils.clip_grad_norm_([prefix], 1.0))
            optimizer.step()
            metrics = {
                "step": step + 1,
                "loss": float(loss.detach()),
                "codec0_loss": float(output.loss.detach()),
                "residual_loss": float(residual_loss.detach()),
                "gradient_norm": grad_norm,
                "elapsed_seconds": time.perf_counter() - started,
            }
            log.write(json.dumps(metrics) + "\n")
            log.flush()
            if (step + 1) % 10 == 0 or step == start:
                print(json.dumps(metrics), flush=True)
            if (step + 1) % 50 == 0 or step + 1 == steps:
                save_checkpoint(folder, prefix, optimizer, step + 1, metadata)
    assert all(p.grad is None for p in wrapper.model.parameters())
    maximum = int(gold.shape[1]) + 24
    trained = render(builder, wrapper, prefix, speaker, folder / "trained.wav", maximum, seed)
    null = render(builder, wrapper, initial, speaker, folder / "null.wav", maximum, seed)
    metadata.update(
        {
            "trained_output": trained,
            "null_output": null,
            "elapsed_train_and_render_seconds": time.perf_counter() - started,
            "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated()
            if device.type == "cuda"
            else None,
            "frozen_model_gradients_absent": True,
            "inference_uses_utterance_text": False,
        }
    )
    (folder / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"trained": trained, "null": null}), flush=True)


def infer(folder: Path, out: Path, max_frames: int, seed: int):
    metadata = json.loads((folder / "config.json").read_text())
    wrapper, builder, speaker = load_backbone(metadata)
    prefix = load_file(str(folder / "prefix.safetensors"))["soft_tokens"].to(speaker.device)
    result = render(builder, wrapper, prefix, speaker, out, max_frames, seed)
    out.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
