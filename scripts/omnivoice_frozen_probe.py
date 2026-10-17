"""Managed, offline OMX-01 frozen probe, adapted from the supplied pinned harness.

Target text is empty in H1/H2; H1 alone uses the explicitly authorized immutable
voice transcript. H3 replaces preparation with raw audio masks. No images,
reader, optimizer, training, ASR, adaptive seeds or model downloads are accepted.
"""

import argparse
import hashlib
import inspect
import json
import os
import random
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from types import MethodType

from latent_diaries.omnivoice_protocol import (
    MODEL_REVISION,
    POSITIVE_SENTENCES,
    SOURCE_COMMIT,
    audit_inputs,
    check_api,
    generation_fingerprint,
    get_plan,
    raw_mask_inputs,
    validate_reuse,
)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["model-snapshot", "reference-wav", "reference-text-file", "output-dir"]:
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--phase", choices=["smoke", "full"], default="smoke")
    p.add_argument("--source-commit", required=True)
    p.add_argument("--model-revision", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    p.add_argument("--skip-existing", action="store_true")
    args = p.parse_args()
    if not Path("/.dockerenv").exists() or args.device != "cuda:0":
        raise RuntimeError("Use the managed CUDA worker, no implicit host/CPU fallback")
    if args.source_commit != SOURCE_COMMIT or args.model_revision != MODEL_REVISION:
        raise RuntimeError("Frozen source/checkpoint pin differs")
    required = [
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "audio_tokenizer/config.json",
        "audio_tokenizer/model.safetensors",
        "audio_tokenizer/LICENSE",
    ]
    if (
        args.model_snapshot.resolve().name != MODEL_REVISION
        or any(not (args.model_snapshot / f).is_file() for f in required)
        or not os.statvfs(args.model_snapshot).f_flag & os.ST_RDONLY
    ):
        raise RuntimeError("Complete pinned snapshot on a readonly mount required")
    reference_text = args.reference_text_file.read_text().strip()
    if not reference_text or len(reference_text) > 1000 or not args.reference_wav.is_file():
        raise RuntimeError(
            "Previously authorized exact voice transcript and WAV required; never infer with ASR"
        )
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    from importlib.metadata import version

    import numpy as np
    import soundfile as sf
    import torch
    from omnivoice import OmniVoice, OmniVoiceGenerationConfig
    from omnivoice.models.omnivoice import GenerationTask
    from safetensors.torch import save_file

    check_api(OmniVoice, GenerationTask)
    checkout = Path("/opt/omnivoice/upstream")
    if (
        subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
        != SOURCE_COMMIT
    ):
        raise RuntimeError("Installed checkout pin differs")
    installed = Path(inspect.getfile(OmniVoice))
    if sha(installed) != sha(checkout / "omnivoice/models/omnivoice.py"):
        raise RuntimeError("Installed model source differs from pinned checkout")
    ref_wave, rate = sf.read(args.reference_wav, always_2d=True)
    if not 3 <= len(ref_wave) / rate <= 10 or not np.isfinite(ref_wave).all():
        raise RuntimeError("Finite consented3–10second reference required")
    cfg = OmniVoiceGenerationConfig(
        num_step=32,
        guidance_scale=2.0,
        position_temperature=5.0,
        class_temperature=0.0,
        postprocess_output=False,
    )
    fingerprint = {
        "source_commit": SOURCE_COMMIT,
        "model_revision": MODEL_REVISION,
        "harness_sha256": sha(__file__),
        "protocol_source_sha256": sha("src/latent_diaries/omnivoice_protocol.py"),
        "upstream_modeling_sha256": sha(installed),
        "generation_config": generation_fingerprint(cfg),
        "reference_sha256": sha(args.reference_wav),
        "reference_text_sha256": sha(args.reference_text_file),
        "worker_lock_sha256": sha("uv.lock"),
        "omnivoice_lock_sha256": sha("/opt/omnivoice/uv.lock"),
        "packages": {
            k: version(k)
            for k in [
                "torch",
                "torchaudio",
                "transformers",
                "huggingface-hub",
                "omnivoice",
                "numpy",
                "safetensors",
            ]
        },
        "torch_cuda": torch.version.cuda,
        "dtype": args.dtype,
        "device": args.device,
        "gpu": torch.cuda.get_device_name(),
        "reference_preprocess_prompt": True,
        "no_training_performed": True,
        "network_disabled": Path("/sys/class/net").exists()
        and all(n.name == "lo" for n in Path("/sys/class/net").iterdir()),
    }
    if not fingerprint["network_disabled"]:
        raise RuntimeError("Use the network-disabled comparator compose override")
    args.output_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    provenance = args.output_dir / "run-provenance.json"
    if provenance.exists():
        if not args.skip_existing or json.loads(provenance.read_text()) != fingerprint:
            raise RuntimeError("Run provenance changed; preserve earlier run")
    else:
        if any(args.output_dir.iterdir()):
            raise RuntimeError("Fresh output or matching provenance required")
        provenance.write_text(json.dumps(fingerprint, indent=2) + "\n")
    run_sha = sha(provenance)
    if args.phase == "full":
        smoke = args.output_dir / "smoke-summary.json"
        if not smoke.is_file() or json.loads(smoke.read_text()).get("technical_pass") is not True:
            raise RuntimeError("Full grid requires a technically completed smoke")
    model = (
        OmniVoice.from_pretrained(
            str(args.model_snapshot.resolve()), device_map=args.device, dtype=getattr(torch, args.dtype)
        )
        .eval()
        .requires_grad_(False)
    )

    def forbidden_asr(*values, **kwargs):
        raise RuntimeError("Reference transcript provided; ASR is forbidden in this probe")

    model.load_asr_model = forbidden_asr
    model.transcribe = forbidden_asr
    with torch.inference_mode():
        prompt = model.create_voice_clone_prompt(
            ref_audio=str(args.reference_wav.resolve()), ref_text=reference_text
        )
    if not prompt.ref_text or prompt.ref_audio_tokens.ndim != 2 or prompt.ref_audio_tokens.shape[0] != 8:
        raise RuntimeError("Fixed voice prompt must have text and8 acoustic streams")
    fixed_prompt_text = prompt.ref_text
    fixed_prompt_codes = prompt.ref_audio_tokens.detach().clone()
    frame_rate = float(model.audio_tokenizer.config.frame_rate)
    if frame_rate <= 0 or model.config.num_audio_codebook != 8:
        raise RuntimeError("Unexpected codec contract")
    index = []
    for mode, seed, seconds, baseline_idx in get_plan(args.phase):
        name = f"{mode}_s{seed}_d{int(seconds)}"
        wav = args.output_dir / (name + ".wav")
        file = args.output_dir / (name + ".json")
        codefile = args.output_dir / (name + ".safetensors")
        if any(f.exists() for f in [wav, file, codefile]):
            if not args.skip_existing or not all(f.exists() for f in [wav, file, codefile]):
                raise RuntimeError("Partial/existing case requires audit: " + name)
            validate_reuse(
                json.loads(file.read_text()), run_sha, mode, seed, seconds, sha(wav), sha(codefile)
            )
            index.append({"case": name, "status": "reused"})
            continue
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()
        calls = []
        captured = []
        cfg_identical = []
        prepare = model._prepare_inference_inputs
        decode = model._decode_and_post_process

        def observe(
            self,
            text,
            num_target_tokens,
            ref_text=None,
            ref_audio_tokens=None,
            lang=None,
            instruct=None,
            denoise=True,
            _mode=mode,
            _prepare=prepare,
            _calls=calls,
        ):
            if _mode in {"positive", "voice-empty"} and (
                ref_text != fixed_prompt_text
                or not torch.equal(ref_audio_tokens, fixed_prompt_codes)
            ):
                raise RuntimeError("Immutable voice prompt changed")
            out = (
                raw_mask_inputs(self, num_target_tokens)
                if _mode == "mask-only"
                else _prepare(
                    text, num_target_tokens, ref_text, ref_audio_tokens, lang, instruct, denoise
                )
            )
            _calls.append(
                audit_inputs(
                    _mode,
                    text,
                    num_target_tokens,
                    ref_text,
                    ref_audio_tokens,
                    lang,
                    instruct,
                    out,
                    self.config.audio_mask_id,
                )
            )
            return out

        def checked_decode(
            self, tokens, rms, gen_config, _seconds=seconds, _captured=captured, _decode=decode
        ):
            if not isinstance(tokens, torch.Tensor) or tokens.shape != (8, round(_seconds * frame_rate)):
                raise RuntimeError("Unexpected decoded code shape")
            if tokens.dtype != torch.long or bool(
                ((tokens < 0) | (tokens >= self.config.audio_mask_id)).any()
            ):
                raise RuntimeError("MASK/out-of-range codes remain at codec boundary")
            _captured.append(tokens.detach().cpu().contiguous())
            return _decode(tokens, rms, gen_config)

        def inspect_forward(module, values, kwargs, _mode=mode, _identical=cfg_identical):
            if _mode == "mask-only":
                ids = kwargs["input_ids"]
                mask = kwargs["audio_mask"]
                attention = kwargs["attention_mask"]
                _identical.append(
                    torch.equal(ids[0], ids[1])
                    and torch.equal(mask[0], mask[1])
                    and torch.equal(attention[0], attention[1])
                )

        model._prepare_inference_inputs = MethodType(observe, model)
        model._decode_and_post_process = MethodType(checked_decode, model)
        handle = model.register_forward_pre_hook(inspect_forward, with_kwargs=True)
        started = time.monotonic()
        try:
            with torch.inference_mode():
                if mode == "mask-only":
                    frames = round(seconds * frame_rate)
                    task = GenerationTask(
                        batch_size=1,
                        texts=[""],
                        target_lens=[frames],
                        langs=[None],
                        instructs=[None],
                        ref_texts=[None],
                        ref_audio_tokens=[None],
                        ref_rms=[None],
                    )
                    codes = model._generate_iterative(task, cfg)[0]
                    audio = model._decode_and_post_process(codes, None, cfg)
                else:
                    kwargs = {
                        "text": POSITIVE_SENTENCES[baseline_idx] if mode == "positive" else "",
                        "language": "it",
                        "duration": seconds,
                        "generation_config": cfg,
                    }
                    if mode in {"positive", "voice-empty"}:
                        kwargs["voice_clone_prompt"] = prompt
                    audio = model.generate(**kwargs)[0]
        except Exception as exc:
            (args.output_dir / (name + "-failure.json")).write_text(
                json.dumps(
                    {
                        "case": name,
                        "error_type": type(exc).__name__,
                        "operational_failure_not_model_quality": True,
                    },
                    indent=2,
                )
                + "\n"
            )
            raise
        finally:
            handle.remove()
            model._prepare_inference_inputs = prepare
            model._decode_and_post_process = decode
        elapsed = time.monotonic() - started
        if (
            len(calls) != 1
            or len(captured) != 1
            or (mode == "mask-only" and (len(cfg_identical) != 32 or not all(cfg_identical)))
        ):
            raise RuntimeError("Assembly/code capture or identical CFG audit failed")
        x = np.asarray(audio, dtype=np.float32)
        if x.ndim != 1 or len(x) < int(model.sampling_rate * 0.1) or not np.isfinite(x).all():
            raise RuntimeError("Empty/nonfinite decoder waveform")
        sf.write(wav, x, int(model.sampling_rate), subtype="PCM_16")
        if sha(wav) == fingerprint["reference_sha256"]:
            raise RuntimeError("Output is byte-identical to reference")
        save_file({"audio_codes": captured[0]}, str(codefile))
        if any(p.requires_grad or p.grad is not None for p in model.parameters()):
            raise RuntimeError("Frozen-only invariant failed")
        rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
        record = {
            "case": name,
            "mode": mode,
            "seed": seed,
            "requested_duration_seconds": seconds,
            "actual_duration_seconds": len(x) / model.sampling_rate,
            "sample_rate": model.sampling_rate,
            "elapsed_seconds": elapsed,
            "peak_vram_bytes": torch.cuda.max_memory_allocated(),
            "run_provenance_sha256": run_sha,
            "wav_sha256": sha(wav),
            "codes_sha256": sha(codefile),
            "frames": captured[0].shape[1],
            "codebooks": 8,
            "frame_rate": frame_rate,
            "waveform_rms": rms,
            "peak_absolute_amplitude": float(np.abs(x).max()),
            "clipped_sample_fraction": float(np.mean(np.abs(x) >= 1.0)),
            "acoustic_silence_candidate": rms < 1e-5,
            "technical_valid": True,
            "input_audit": calls[0],
            "mask_only_cfg_inputs_identical": all(cfg_identical) if cfg_identical else None,
            "textless_target": mode != "positive",
            "fixed_voice_reference_used": mode in {"positive", "voice-empty"},
            "human_assessment": "pending",
            "generation_config": asdict(cfg),
        }
        file.write_text(json.dumps(record, indent=2) + "\n")
        index.append({"case": name, "status": "generated"})
        print(
            json.dumps({"completed": name, "seconds": record["actual_duration_seconds"], "rms": rms}),
            flush=True,
        )
    (args.output_dir / (args.phase + "-summary.json")).write_text(
        json.dumps(
            {
                "phase": args.phase,
                "cases": index,
                "run_provenance_sha256": run_sha,
                "technical_pass": True,
                "no_training_performed": True,
                "human_gate_passed": False,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
