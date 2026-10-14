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


def frame_logits(logits, prefix_length: int, frames: int):
    """Align frame zero at the final codec BOS; include the EOS prediction."""
    if prefix_length < 1 or frames < 1 or logits.shape[1] < prefix_length + frames:
        raise ValueError("Invalid prefix/frame alignment")
    return logits[:, prefix_length - 1 : prefix_length + frames]


def distribution_kl(student, teacher):
    """Mean KL per aligned frame/codebook; teacher is supervision only."""
    if student.shape != teacher.shape:
        raise ValueError("Teacher/student prediction shapes must match after frame alignment")
    return (
        torch.nn.functional.kl_div(
            torch.log_softmax(student.float(), dim=-1),
            torch.log_softmax(teacher.detach().float(), dim=-1),
            log_target=True,
            reduction="none",
        )
        .sum(-1)
        .mean()
    )


def trajectory_logits(wrapper, builder, prefix, speaker, codes):
    """Policies conditioned on the same previous actions, with aligned heads."""
    output = wrapper.model.talker(
        **builder.build_teacher_forced(condition(prefix, speaker), codes), output_hidden_states=True
    )
    length = builder.build_prefill(condition(prefix, speaker))["inputs_embeds"].shape[1]
    hidden = output.hidden_states[0][-1][0, length - 1 : length - 1 + codes.shape[1]]
    return frame_logits(output.logits, length, codes.shape[1]), aligned_residual_logits(
        wrapper.model.talker, hidden, codes[0]
    )


@torch.no_grad()
def render(
    builder, wrapper, prefix, speaker, out: Path, max_frames: int, seed: int,
    *, codec0_greedy: bool = False,
):
    import soundfile as sf

    codes = builder.generate_codes(
        condition(prefix, speaker), max_frames=max_frames, seed=seed, codec0_greedy=codec0_greedy
    )[0]
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
        "codec0_greedy": codec0_greedy,
        "residuals_sampled": True,
        "seed": seed,
        "max_frames": max_frames,
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
    teacher_cache_dir: Path | None = None,
    cached_gold_ce: bool = False,
    init_prefix_dir: Path | None = None,
    rollout_every: int = 0,
    rollout_weight: float = 1.0,
):
    if steps < 1 or tokens < 1 or learning_rate <= 0 or residual_loss_weight < 0:
        raise ValueError("steps, tokens and learning rate must be positive")
    baseline = json.loads(Path("artifacts/m0_italian.json").read_text())
    teacher_info = None
    if cached_gold_ce and teacher_cache_dir is None:
        raise ValueError("Cached-gold CE requires a teacher cache")
    if rollout_every < 0 or rollout_weight <= 0:
        raise ValueError("Invalid rollout refresh/weight")
    if rollout_every and (teacher_cache_dir is None or cached_gold_ce):
        raise ValueError("Rollout supervision requires teacher-distribution KL")
    if teacher_cache_dir is not None:
        if residual_loss_weight <= 0:
            raise ValueError("Teacher distribution distillation requires residual supervision")
        teacher_info = json.loads((teacher_cache_dir / "config.json").read_text())
        if digest(target) != teacher_info["target_audio_sha256"]:
            raise ValueError("Target audio differs from the cached teacher trajectory")
        for key in ["model_id", "model_revision", "lock_sha256"]:
            expected = digest(Path("uv.lock")) if key == "lock_sha256" else baseline[key]
            if teacher_info[key] != expected:
                raise ValueError(f"Teacher cache configuration differs: {key}")
        if digest(teacher_cache_dir / "teacher.safetensors") != teacher_info["teacher_cache_sha256"]:
            raise ValueError("Teacher cache checksum differs")
    metadata = {
        "stage": "M1.1 single-utterance inversion control",
        "model_id": baseline["model_id"],
        "model_revision": baseline["model_revision"],
        "seed": seed,
        "tokens": tokens,
        "learning_rate": learning_rate,
        "precision": "float32",
        "residual_loss_weight": residual_loss_weight,
        "objective": "teacher_distribution_kl"
        if teacher_info and not cached_gold_ce
        else "gold_codec_ce",
        "teacher_cache_sha256": teacher_info["teacher_cache_sha256"] if teacher_info else None,
        "init_prefix_sha256": digest(init_prefix_dir / "prefix.safetensors")
        if init_prefix_dir
        else None,
        "rollout_every": rollout_every,
        "rollout_weight": rollout_weight,
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
            "objective",
            "teacher_cache_sha256",
            "init_prefix_sha256",
            "rollout_every",
            "rollout_weight",
            "target_sha256",
            "lock_sha256",
            "source_sha256",
        ]:
            if metadata[key] != saved.get(key):
                raise ValueError(f"Resume configuration changed: {key}")
        metadata = saved
    torch.manual_seed(seed)
    wrapper, builder, speaker = load_backbone(metadata)
    device = speaker.device
    teacher = None
    if teacher_cache_dir is not None:
        teacher = {
            k: v.to(device) for k, v in load_file(str(teacher_cache_dir / "teacher.safetensors")).items()
        }
        if list(builder.role_ids) != teacher_info["role_ids"]:
            raise ValueError("Teacher/student fixed-role protocol differs")
    with torch.no_grad():
        gold = (
            teacher["audio_codes"]
            if teacher
            else wrapper.model.speech_tokenizer.encode(str(target)).audio_codes[0][None].to(device)
        )
        pad_id = torch.tensor([[wrapper.model.config.tts_pad_token_id]], device=device)
        pad = wrapper.model.talker.text_projection(wrapper.model.talker.get_text_embeddings()(pad_id))
        initial = pad.expand(1, tokens, -1).clone() + 0.02 * torch.randn(
            1, tokens, pad.shape[-1], device=device
        )
    torch.save(gold.cpu(), folder / "gold_codes.pt")
    prefix = torch.nn.Parameter(initial.clone())
    if init_prefix_dir is not None:
        initialized = load_file(str(init_prefix_dir / "prefix.safetensors"))["soft_tokens"].to(device)
        if initialized.shape != prefix.shape:
            raise ValueError("Initial checkpoint token/hidden shape differs")
        prefix.data.copy_(initialized)
    optimizer = torch.optim.AdamW([prefix], lr=learning_rate, weight_decay=0.0)
    teacher_prefix = (
        load_file(str(teacher_cache_dir / "prefix.safetensors"))["soft_tokens"].to(device)
        if rollout_every
        else None
    )
    rollout_cache = None
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
        if rollout_every:
            path = folder / metadata["rollout_cache_file"]
            if digest(path) != metadata["rollout_cache_sha256"]:
                raise ValueError("Rollout cache checksum differs")
            rollout_cache = {k: v.to(device) for k, v in load_file(str(path)).items()}
    metadata["trainable_parameters"] = prefix.numel()
    metadata["target_frames"] = gold.shape[1]
    metadata["role_ids"] = list(builder.role_ids)
    started = time.perf_counter()
    with (folder / "loss.jsonl").open("a") as log:
        for step in range(start, steps):
            if rollout_every and (rollout_cache is None or step % rollout_every == 0):
                sampled = builder.generate_codes(
                    condition(prefix, speaker),
                    max_frames=int(gold.shape[1]) + 24,
                    seed=seed + step // rollout_every,
                )[0][None]
                if not sampled.shape[1]:
                    raise RuntimeError("Empty on-policy history")
                with torch.no_grad():
                    teacher0, teacher_rest = trajectory_logits(
                        wrapper, builder, teacher_prefix, speaker, sampled
                    )
                rollout_cache = {
                    "audio_codes": sampled.detach(),
                    "codec0_logits": teacher0.detach(),
                    "residual_logits": teacher_rest.detach(),
                }
                filename = f"rollout-step-{step:06d}.safetensors"
                save_file({k: v.cpu() for k, v in rollout_cache.items()}, str(folder / filename))
                metadata["rollout_cache_file"] = filename
                metadata["rollout_cache_sha256"] = digest(folder / filename)
                print(
                    json.dumps({"rollout_refreshed_at_step": step, "frames": sampled.shape[1]}),
                    flush=True,
                )
            optimizer.zero_grad(set_to_none=True)
            batch = builder.build_teacher_forced(condition(prefix, speaker), gold)
            output = wrapper.model.talker(**batch, output_hidden_states=residual_loss_weight > 0)
            loss = output.loss
            residual_loss = torch.zeros((), device=device)
            codec0_kl = torch.zeros((), device=device)
            residual_kl = torch.zeros((), device=device)
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
                if teacher is not None:
                    codec0_kl = distribution_kl(
                        frame_logits(output.logits, prefix_length, gold.shape[1]),
                        teacher["codec0_logits"],
                    )
                    residual_kl = distribution_kl(residual_logits, teacher["residual_logits"])
                    if not cached_gold_ce:
                        loss = codec0_kl + residual_loss_weight * residual_kl
            rollout_loss = torch.zeros((), device=device)
            if rollout_cache is not None:
                live0, live_rest = trajectory_logits(
                    wrapper, builder, prefix, speaker, rollout_cache["audio_codes"]
                )
                rollout_loss = distribution_kl(
                    live0, rollout_cache["codec0_logits"]
                ) + residual_loss_weight * distribution_kl(live_rest, rollout_cache["residual_logits"])
                loss = (loss + rollout_weight * rollout_loss) / (1 + rollout_weight)
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
                "codec0_kl": float(codec0_kl.detach()),
                "residual_kl": float(residual_kl.detach()),
                "rollout_kl": float(rollout_loss.detach()),
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


def infer(folder: Path, out: Path, max_frames: int, seed: int, codec0_greedy: bool = False):
    metadata = json.loads((folder / "config.json").read_text())
    wrapper, builder, speaker = load_backbone(metadata)
    prefix = load_file(str(folder / "prefix.safetensors"))["soft_tokens"].to(speaker.device)
    result = render(
        builder, wrapper, prefix, speaker, out, max_frames, seed, codec0_greedy=codec0_greedy
    )
    out.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
