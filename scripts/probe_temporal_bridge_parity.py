"""Isolate official versus exact continuous conditioning under both decoders."""

import argparse
import json
import os
from pathlib import Path

import soundfile as sf
import torch
from safetensors.torch import save_file

from latent_diaries.inversion import condition, digest, load_backbone


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--sequence-id", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    if not Path("/.dockerenv").exists() or args.output_dir.exists():
        raise ValueError("Managed worker and fresh isolated control directory required")
    row = next(r for r in json.loads(args.corpus.read_text())["rows"] if r["id"] == args.sequence_id)
    wrapper, builder, speaker = load_backbone(json.loads(Path("artifacts/m0_italian.json").read_text()))
    args.output_dir.mkdir(mode=0o700)
    prompt = wrapper.create_voice_clone_prompt(
        ref_audio=os.environ["LD_REF_AUDIO"], x_vector_only_mode=True
    )
    text = row["fragments"][1]
    with torch.no_grad():
        ids = wrapper._tokenize_texts([wrapper._build_assistant_text(text)])[0][:, 3:-5]
        prefix = wrapper.model.talker.text_projection(wrapper.model.talker.get_text_embeddings()(ids))
        visual = condition(prefix, speaker)
        prefill = builder.build_prefill(visual)
    records = []
    for greedy in [True, False]:
        captured = {}
        original = wrapper.model.talker.generate

        def capture(**kwargs):
            for key, value in prefill.items():
                torch.testing.assert_close(value, kwargs[key], rtol=0, atol=0)
            result = original(**kwargs)
            frames = [h[-1] for h in result.hidden_states if h[-1] is not None]
            codes = torch.stack(frames, dim=1)[0]
            eos = wrapper.model.config.talker_config.codec_eos_token_id
            stop = (codes[:, 0] == eos).nonzero()
            captured["codes"] = codes[: int(stop[0]) if len(stop) else len(codes)][:240].cpu()
            return result

        wrapper.model.talker.generate = capture
        try:
            with torch.inference_mode(), torch.random.fork_rng(devices=[speaker.device.index or 0]):
                torch.manual_seed(42)
                waves, rate = wrapper.generate_voice_clone(
                    text=text,
                    language="Italian",
                    voice_clone_prompt=prompt,
                    non_streaming_mode=True,
                    max_new_tokens=241,
                    min_new_tokens=2,
                    do_sample=not greedy,
                    top_k=50,
                    top_p=1.0,
                    temperature=0.9,
                    repetition_penalty=1.05,
                    subtalker_dosample=True,
                    subtalker_top_k=50,
                    subtalker_top_p=1.0,
                    subtalker_temperature=0.9,
                )
        finally:
            wrapper.model.talker.generate = original
        mode = "greedy" if greedy else "sampled"
        file = args.output_dir / (mode + "-official.wav")
        sf.write(file, waves[0], rate)
        with torch.inference_mode():
            codes = builder.generate_codes(visual, max_frames=240, seed=42, codec0_greedy=greedy)[
                0
            ].cpu()
        equal = torch.equal(codes, captured["codes"])
        save_file(
            {"official": captured["codes"], "continuous": codes},
            str(args.output_dir / (mode + ".safetensors")),
        )
        record = {
            "mode": mode,
            "prefill_exact": True,
            "codes_exact": equal,
            "frames": len(codes),
            "official_frames": len(captured["codes"]),
            "reached_cap": len(codes) >= 240,
            "audio": str(file),
            "audio_sha256": digest(file),
        }
        records.append(record)
        print(json.dumps(record), flush=True)
        if not equal:
            raise RuntimeError("Official/continuous autoregression differs: inspect saved tensors")
    (args.output_dir / "report.json").write_text(
        json.dumps(
            {
                "scope": "Offline text-conditioned parity diagnosis",
                "source_sha256": digest(Path(__file__)),
                "corpus_sha256": digest(args.corpus),
                "records": records,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
