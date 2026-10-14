"""Local auxiliary intelligibility probe; never a speech/artistic training loss."""

import argparse
import json
from pathlib import Path

import librosa
import soundfile as sf
import torch
from huggingface_hub import HfApi, snapshot_download
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision")
    args = parser.parse_args()
    if not Path("/.dockerenv").exists():
        raise RuntimeError("Use the managed worker")
    model_id = "openai/whisper-tiny"
    revision = HfApi().model_info(model_id, revision=args.revision or "main").sha
    snapshot = snapshot_download(
        model_id, revision=revision, allow_patterns=["*.json", "*.txt", "model.safetensors"]
    )
    processor = WhisperProcessor.from_pretrained(snapshot)
    model = WhisperForConditionalGeneration.from_pretrained(snapshot).to("cuda:0").eval()
    results = []
    for path in args.audio:
        audio, sr = sf.read(path)
        if audio.ndim > 1:
            audio = audio.mean(-1)
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        inputs = processor(
            audio, sampling_rate=16000, return_tensors="pt", return_attention_mask=True
        ).to(model.device)
        with torch.inference_mode():
            tokens = model.generate(
                **inputs, language="it", task="transcribe", do_sample=False, max_new_tokens=128
            )
        row = {
            "audio": str(path),
            "auxiliary_transcription": processor.batch_decode(tokens, skip_special_tokens=True)[0],
        }
        results.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    result = {
        "scope": "Auxiliary ASR diagnostic, not human judgment or artistic metric",
        "model_id": model_id,
        "model_revision": revision,
        "forced_language": "Italian",
        "audio_uploaded": False,
        "results": results,
    }
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
