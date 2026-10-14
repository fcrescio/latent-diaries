"""Synthetic demonstration inputs only; never a claim of generated model output."""
from __future__ import annotations
import json
import math
import struct
import wave
from pathlib import Path
from PIL import Image, ImageDraw
from .review import ingest


def create_demo(root: Path, db_path: Path, assets: Path) -> int:
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "demo.jsonl"
    with manifest.open("w", encoding="utf8") as stream:
        for n in range(3):
            image = Image.new("RGB", (480, 320), color=(27 + n*20, 47+n*10, 72+n*15))
            draw = ImageDraw.Draw(image)
            draw.rectangle([130, 130, 350, 270], fill=(100, 120, 150))
            draw.ellipse([200+n*20, 25, 280+n*20, 105], fill=(235, 207, 134))
            image_path = root / f"synthetic-{n}.png"
            image.save(image_path)
            audio_path = root / f"synthetic-{n}.wav"
            sr = 16000
            with wave.open(str(audio_path), "wb") as wav:
                wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(sr)
                frames = bytearray()
                for t in range(sr * 2):
                    value = int(6000*math.sin(2*math.pi*(260+n*85)*t/sr))
                    frames.extend(struct.pack("<h", value))
                wav.writeframes(bytes(frames))
            stream.write(json.dumps({"id": f"synthetic-demo-{n}", "run_id": "DEMO-NON-ML",
                                     "blind_label": "synthetic", "image": image_path.name,
                                     "audio": audio_path.name}) + "\n")
    return ingest(manifest, db_path, assets)
