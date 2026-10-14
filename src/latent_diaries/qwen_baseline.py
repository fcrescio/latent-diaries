"""M0: genuine Italian Qwen TTS baseline using user's consented reference audio."""
from __future__ import annotations
import os
import hashlib
import json
import time
import subprocess
from importlib.metadata import version
from pathlib import Path

DEFAULT_TEXT = "La città sembrava trattenere il respiro, come se aspettasse una risposta."


def run(out: Path, text: str = DEFAULT_TEXT) -> None:
    ref = os.getenv("LD_REF_AUDIO", "")
    if not ref or not Path(ref).is_file():
        raise RuntimeError("Set LD_REF_AUDIO to a local consenting speaker's WAV; mounted inside container.")
    import torch
    import soundfile as sf
    from qwen_tts import Qwen3TTSModel
    from huggingface_hub import HfApi, snapshot_download

    started = time.perf_counter()
    model_id = os.getenv("LD_QWEN_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    revision = HfApi().model_info(model_id, revision=os.getenv("LD_QWEN_REVISION") or "main").sha
    # Upstream wrapper doesn't propagate revision to processor/codec loads.
    # A complete immutable local snapshot pins all components together.
    snapshot = snapshot_download(model_id, revision=revision)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device.startswith("cuda") and torch.cuda.is_bf16_supported() else torch.float32
    model = Qwen3TTSModel.from_pretrained(
        snapshot,
        device_map=device, dtype=dtype, attn_implementation="sdpa")
    loaded = time.perf_counter()
    torch.manual_seed(42)
    args = dict(text=text, language="Italian", ref_audio=ref, non_streaming_mode=True)
    ref_text = os.getenv("LD_REF_TEXT", "").strip()
    if ref_text:
        args["ref_text"] = ref_text
    else:
        args["x_vector_only_mode"] = True
    wavs, sr = model.generate_voice_clone(**args)
    generated = time.perf_counter()
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out, wavs[0], sr)
    encoded = model.model.speech_tokenizer.encode(ref)
    codes = encoded.audio_codes[0]
    if codes.ndim != 2 or codes.shape[1] != 16:
        raise RuntimeError("Expected genuine speech codec output [frames,16]")
    roundtrip, codec_sr = model.model.speech_tokenizer.decode(encoded)
    roundtrip_path = out.with_name(out.stem + "_codec_roundtrip.wav")
    sf.write(roundtrip_path, roundtrip[0], codec_sr)
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    manifest = dict(stage="M0", model_id=model_id, model_revision=revision,
        seed=42, device=device, dtype=str(dtype),
        gpu=torch.cuda.get_device_name() if device.startswith("cuda") else None,
        peak_vram_bytes=torch.cuda.max_memory_allocated() if device.startswith("cuda") else None,
        load_seconds=loaded-started, generate_seconds=generated-loaded,
        sample_rate=sr, duration_seconds=len(wavs[0])/sr,
        codec_sample_rate=codec_sr, codec_shape=list(codes.shape),
        reference_sha256=hashlib.sha256(Path(ref).read_bytes()).hexdigest(),
        output_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
        code_revision=os.getenv("LD_CODE_REVISION") or (git.stdout.strip() if git.returncode == 0 else None),
        lock_sha256=hashlib.sha256(Path("uv.lock").read_bytes()).hexdigest(),
        packages={p: version(p) for p in ["torch", "transformers", "qwen-tts", "huggingface-hub"]},
        human_listening_status="pending", text_conditioned_baseline=True)
    out.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved genuine Qwen baseline to {out}")
