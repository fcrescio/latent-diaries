"""Training-only teacher control: exact stock codes and frame-aligned logits.

Teacher text is allowed only here. A cached teacher-derived prefix is a positive
protocol control, not a learned textless or image-conditioned model result.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import soundfile as sf
import torch
from safetensors.torch import save_file

from latent_diaries.inversion import aligned_residual_logits, condition, digest, load_backbone
from latent_diaries.qwen_baseline import DEFAULT_TEXT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    folder = args.output_dir
    if folder.exists() and any(folder.iterdir()):
        raise ValueError("Use a fresh teacher-cache directory")
    folder.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(Path("artifacts/m0_italian.json").read_text())
    wrapper, builder, speaker = load_backbone(metadata)
    prompt = wrapper.create_voice_clone_prompt(
        ref_audio=os.environ["LD_REF_AUDIO"], x_vector_only_mode=True
    )
    ids = wrapper._tokenize_texts([wrapper._build_assistant_text(DEFAULT_TEXT)])[0]
    with torch.no_grad():
        prefix = wrapper.model.talker.text_projection(
            wrapper.model.talker.get_text_embeddings()(ids[:, 3:-5])
        )
    captured = {}
    original = wrapper.model.generate

    def capture(**kwargs):
        result = original(**kwargs)
        captured["codes"] = result[0][0]
        return result

    wrapper.model.generate = capture
    try:
        with torch.random.fork_rng(devices=[speaker.device.index or 0]):
            torch.manual_seed(42)
            wavs, sr = wrapper.generate_voice_clone(
                text=DEFAULT_TEXT,
                language="Italian",
                voice_clone_prompt=prompt,
                non_streaming_mode=True,
                max_new_tokens=93,
                do_sample=True,
                top_k=50,
                top_p=1.0,
                temperature=0.9,
                subtalker_dosample=True,
                subtalker_top_k=50,
                subtalker_top_p=1.0,
                subtalker_temperature=0.9,
                repetition_penalty=1.05,
            )
    finally:
        wrapper.model.generate = original
    sf.write(folder / "teacher.wav", wavs[0], sr)
    gold = captured["codes"][None]
    teacher_condition = condition(prefix, speaker)
    with torch.no_grad():
        packed = builder.build_teacher_forced(teacher_condition, gold)
        out = wrapper.model.talker(**packed, output_hidden_states=True)
        length = builder.build_prefill(teacher_condition)["inputs_embeds"].shape[1]
        first = out.logits[:, length - 1 : length + gold.shape[1]].float()
        hidden = out.hidden_states[0][-1][0, length - 1 : length - 1 + gold.shape[1]]
        residual = aligned_residual_logits(wrapper.model.talker, hidden, gold[0]).float()
        assert first.shape[1] == gold.shape[1] + 1
    save_file({"soft_tokens": prefix.detach().cpu()}, str(folder / "prefix.safetensors"))
    save_file(
        {"audio_codes": gold.cpu(), "codec0_logits": first.cpu(), "residual_logits": residual.cpu()},
        str(folder / "teacher.safetensors"),
    )
    result = {
        "stage": "Training-only stock x-vector teacher cache",
        "model_id": metadata["model_id"],
        "model_revision": metadata["model_revision"],
        "seed": 42,
        "tokens": prefix.shape[1],
        "role_ids": list(builder.role_ids),
        "target_frames": gold.shape[1],
        "target_audio_sha256": digest(folder / "teacher.wav"),
        "training_text_sha256": hashlib.sha256(DEFAULT_TEXT.encode()).hexdigest(),
        "teacher_cache_sha256": digest(folder / "teacher.safetensors"),
        "lock_sha256": digest(Path("uv.lock")),
        "source_sha256": digest(Path(__file__)),
        "teacher_codec0_ce": float(out.loss),
        "sample_rate": sr,
        "seconds": len(wavs[0]) / sr,
        "linguistic_conditioning": "Projected training text; positive control, not learned speech",
        "gold_source": "Exact generated codes, not waveform re-encoding",
    }
    (folder / "config.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
