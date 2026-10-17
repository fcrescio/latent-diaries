"""Frozen OMX-01 case plan and conditioning assertions; no model dependency."""

import inspect
from dataclasses import asdict

SOURCE_COMMIT = "08be0b4ccbac3e13e374e86fbfead4b4cac343e2"
MODEL_REVISION = "c5fdb5ccb189668d56333f77ba2629f4cd7535f4"
TEXTLESS_MODES = ("voice-empty", "lang-empty", "mask-only")
POSITIVE_SENTENCES = (
    "La sera il vento muoveva appena le foglie sotto i lampioni.",
    "Avevo quasi dimenticato quella strada, eppure mi sembrava di conoscerla.",
)


def get_plan(phase):
    positive = [("positive", 41, 4.0, 0), ("positive", 42, 6.0, 1)]
    if phase == "smoke":
        return [positive[0]] + [(mode, 41, 4.0, None) for mode in TEXTLESS_MODES]
    if phase != "full":
        raise ValueError("Unknown frozen phase")
    return positive + [
        (mode, seed, duration, None)
        for mode in TEXTLESS_MODES
        for duration in (4.0, 6.0)
        for seed in (41, 42, 43)
    ]


def check_api(model_type, task_type):
    expected = {
        "_prepare_inference_inputs": [
            "self",
            "text",
            "num_target_tokens",
            "ref_text",
            "ref_audio_tokens",
            "lang",
            "instruct",
            "denoise",
        ],
        "_generate_iterative": ["self", "task", "gen_config"],
        "_decode_and_post_process": ["self", "tokens", "rms", "gen_config"],
    }
    for name, parameters in expected.items():
        if list(inspect.signature(getattr(model_type, name)).parameters) != parameters:
            raise RuntimeError("Pinned private API signature changed: " + name)
    required = {
        "batch_size",
        "texts",
        "target_lens",
        "langs",
        "instructs",
        "ref_texts",
        "ref_audio_tokens",
        "ref_rms",
    }
    if not required <= set(inspect.signature(task_type).parameters):
        raise RuntimeError("Pinned GenerationTask signature changed")


def raw_mask_inputs(model, num_target_tokens):
    import torch

    if (
        type(num_target_tokens) is not int
        or num_target_tokens < 1
        or model.config.num_audio_codebook != 8
    ):
        raise ValueError("Explicit positive length and8 codebooks required")
    return {
        "input_ids": torch.full(
            (1, 8, num_target_tokens), model.config.audio_mask_id, dtype=torch.long, device=model.device
        ),
        "audio_mask": torch.ones((1, num_target_tokens), dtype=torch.bool, device=model.device),
    }


def audit_inputs(
    mode, text, num_target_tokens, ref_text, ref_audio_tokens, lang, instruct, out, mask_id
):
    import torch

    if mode != "positive" and text != "":
        raise RuntimeError("Target text in textless arm")
    if mode == "mask-only":
        if text or ref_text or ref_audio_tokens is not None or lang or instruct:
            raise RuntimeError("Conditioning leaked into raw mask arm")
        if (
            out["input_ids"].shape != (1, 8, num_target_tokens)
            or out["input_ids"].dtype != torch.long
            or out["audio_mask"].dtype != torch.bool
            or not bool(out["audio_mask"].all())
            or not bool((out["input_ids"] == mask_id).all())
        ):
            raise RuntimeError("Raw mask contract differs")
    elif mode == "lang-empty" and (ref_text or ref_audio_tokens is not None or instruct):
        raise RuntimeError("Speaker/instruction leaked into language-only arm")
    elif mode in {"voice-empty", "positive"} and (not ref_text or ref_audio_tokens is None or instruct):
        raise RuntimeError("Fixed voice reference missing or extra instruction present")
    return {
        "target_text_nonempty": bool(text),
        "reference_transcript_present": bool(ref_text),
        "reference_audio_present": ref_audio_tokens is not None,
        "language": lang,
        "conditional_tokens": out["input_ids"].shape[-1],
        "masked_target_tokens": num_target_tokens,
        "non_audio_positions": int((~out["audio_mask"]).sum()),
        "raw_all_mask": mode == "mask-only",
    }


def generation_fingerprint(config):
    value = asdict(config)
    fixed = {
        "num_step": 32,
        "guidance_scale": 2.0,
        "position_temperature": 5.0,
        "class_temperature": 0.0,
        "postprocess_output": False,
    }
    if any(value.get(k) != v for k, v in fixed.items()):
        raise ValueError("Frozen generation recipe changed")
    return value


def validate_reuse(record, run_sha, mode, seed, seconds, wav_sha, codes_sha):
    expected = {
        "run_provenance_sha256": run_sha,
        "mode": mode,
        "seed": seed,
        "requested_duration_seconds": seconds,
        "wav_sha256": wav_sha,
        "codes_sha256": codes_sha,
    }
    if any(record.get(k) != v for k, v in expected.items()) or record.get("technical_valid") is not True:
        raise RuntimeError("Existing case provenance or checksums changed")
