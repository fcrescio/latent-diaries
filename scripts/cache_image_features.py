"""Cache image-only 768D vectors using the isolated vision runtime in one worker."""
import argparse
import inspect
import json
import os
import time
from pathlib import Path

import torch
import transformers
from PIL import Image, ImageOps
from safetensors.torch import save_file

from latent_diaries.vision import MODEL_ID, FrozenImageEncoder, read_image_manifest, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    if not Path("/.dockerenv").exists():
        raise SystemExit("Run in the managed Docker worker")
    rows = read_image_manifest(args.manifest)
    if args.output_dir.exists():
        raise SystemExit("Use a fresh feature output directory")
    start = time.monotonic()
    encoder = FrozenImageEncoder(args.revision, args.device)
    load_seconds = time.monotonic() - start
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    vectors, controls, records = [], {}, []
    for i, row in enumerate(rows):
        path = Path(row["image"])
        with Image.open(path) as original:
            image = ImageOps.exif_transpose(original).convert("RGB")
        start = time.monotonic()
        vector, shape_info = encoder.encode([image])
        if args.device == "cuda":
            torch.cuda.synchronize()
        record = row | {"image_sha256": sha256(path), "seconds": time.monotonic() - start,
                        "image_size": list(image.size), "processor": shape_info}
        records.append(record)
        vectors.append(vector)
        if i == 0:
            repeated, _ = encoder.encode([image])
            batch, _ = encoder.encode([image, image])
            controls = {
                "repeat_max_abs_error": (vector - repeated).abs().max().item(),
                "batch_single_max_abs_error": (vector - batch).abs().max().item(),
            }
            if controls["repeat_max_abs_error"] > 1e-6 or controls["batch_single_max_abs_error"] > 1e-4:
                raise ValueError("Frozen encoder repeat/batch parity failed")
        print(f"Encoded image {i + 1}/{len(rows)}", flush=True)
    values = torch.cat(vectors)
    assert not any(p.requires_grad for p in encoder.model.parameters())
    args.output_dir.mkdir(parents=True)
    weights = args.output_dir / "features.safetensors"
    save_file({"embeddings": values}, str(weights))
    manifest = {
        "model_id": MODEL_ID, "model_revision": args.revision,
        "transformers_version": transformers.__version__, "torch_version": torch.__version__,
        "device": args.device, "dtype": "float32", "audio_encoder_loaded": False,
        "vision_encoder_loaded": True, "all_parameters_frozen": True,
        "variable_text_used": False, "pooling": "attention-masked mean of final 768D states then L2",
        "shape": list(values.shape), "norms": values.norm(dim=-1).tolist(),
        "cosine_matrix": (values @ values.T).tolist(), "controls": controls,
        "load_seconds": load_seconds,
        "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else None,
        "features_sha256": sha256(weights), "input_manifest_sha256": sha256(args.manifest),
        "code_revision": os.getenv("LD_CODE_REVISION"), "records": records,
        "source_sha256": sha256(Path(__file__)),
        "encoder_source_sha256": sha256(Path(inspect.getfile(FrozenImageEncoder))),
        "upstream_model_source_sha256": sha256(Path(inspect.getfile(type(encoder.model)))),
        "vision_lock_sha256": sha256(Path("/opt/vision/uv.lock")),
        "worker_lock_sha256": sha256(Path("uv.lock")),
        "image_grounding_verified": False, "speech_generated": False,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Feature cache saved; this is not image-to-speech acceptance")


if __name__ == "__main__":
    main()
