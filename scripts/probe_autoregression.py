"""Frozen-checkpoint decoding ablations and an explicitly oracle-only control.

Oracle residuals consume gold target audio codes; they are diagnostic, not
textless generation evidence, and no oracle waveform is saved.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import soundfile as sf
import torch
from safetensors.torch import load_file

from latent_diaries.inversion import condition, digest, load_backbone


def compare(codes, gold):
    frames = min(len(codes), len(gold))
    matches = codes[:frames] == gold[:frames]
    first_wrong = (~matches[:, 0]).nonzero()
    return {
        "frames": len(codes),
        "target_frames": len(gold),
        "aligned_codec0_accuracy": float(matches[:, 0].float().mean()) if frames else None,
        "aligned_residual_accuracy": float(matches[:, 1:].float().mean()) if frames else None,
        "first_codec0_difference_frame": int(first_wrong[0]) if len(first_wrong) else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    metadata = json.loads((args.run_dir / "config.json").read_text())
    wrapper, builder, speaker = load_backbone(metadata)
    talker = wrapper.model.talker
    prefix = load_file(str(args.run_dir / "prefix.safetensors"))["soft_tokens"].to(speaker.device)
    gold = torch.load(args.run_dir / "gold_codes.pt", map_location=speaker.device, weights_only=True)[0]
    original_generate = talker.generate
    original_residual = talker.code_predictor.generate
    results = {}
    settings = [
        ("sampled", True, True, False),
        ("codec0_greedy", False, True, False),
        ("greedy", False, False, False),
        ("oracle_residuals", False, False, True),
    ]
    for name, sample0, sample_rest, oracle in settings:
        cursor = [0]

        def generate(_sample0=sample0, _sample_rest=sample_rest, **kwargs):
            kwargs.update(do_sample=_sample0, subtalker_dosample=_sample_rest)
            return original_generate(**kwargs)

        def residual(_cursor=cursor, **kwargs):
            frame = _cursor[0]
            _cursor[0] += 1
            if frame < len(gold):
                return SimpleNamespace(sequences=gold[frame : frame + 1, 1:])
            return original_residual(**kwargs)

        talker.generate = generate
        if oracle:
            talker.code_predictor.generate = residual
        try:
            codes = builder.generate_codes(
                condition(prefix, speaker), max_frames=len(gold) + 24, seed=metadata["seed"]
            )[0]
        finally:
            talker.generate = original_generate
            talker.code_predictor.generate = original_residual
        result = compare(codes, gold)
        result.update(
            codec0_sampled=sample0,
            residuals_sampled=sample_rest,
            gold_residuals_injected=oracle,
            seed=metadata["seed"],
        )
        if not oracle and len(codes):
            out = args.run_dir / ("probe_" + name + ".wav")
            wavs, sr = wrapper.model.speech_tokenizer.decode([{"audio_codes": codes}])
            sf.write(out, wavs[0], sr)
            result.update(
                audio=str(out), sample_rate=sr, seconds=len(wavs[0]) / sr, audio_sha256=digest(out)
            )
        results[name] = result
        print(json.dumps({name: result}), flush=True)
    packed = builder.build_teacher_forced(condition(prefix, speaker), gold[None])
    with torch.no_grad():
        output = talker(**packed)
    length = builder.build_prefill(condition(prefix, speaker))["inputs_embeds"].shape[1]
    accuracy = (
        (output.logits[0, length - 1 : length - 1 + len(gold)].argmax(-1) == gold[:, 0]).float().mean()
    )
    results["teacher_forced"] = {"codec0_loss": float(output.loss), "codec0_accuracy": float(accuracy)}
    (args.run_dir / "autoregression-probes.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
