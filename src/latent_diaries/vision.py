"""Frozen EmbeddingGemma 2 global vectors; run in the worker's vision interpreter."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

MODEL_ID = "google/embeddinggemma-2"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_image_manifest(path: Path) -> list[dict]:
    """Accept media/provenance only, never captions or linguistic conditions."""
    rows = json.loads(path.read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError("Image manifest must be a nonempty list")
    required = {"id", "image", "scene_group", "source", "license"}
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != required:
            raise ValueError("Only id/image/scene_group/source/license are allowed")
        if not all(isinstance(value, str) and value.strip() for value in row.values()):
            raise ValueError("Image/provenance fields must be nonempty strings")
        if row["id"] in seen:
            raise ValueError("Duplicate image id")
        seen.add(row["id"])
        if not Path(row["image"]).is_file():
            raise ValueError("Missing local image file")
    return rows


def pool_global(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Official final 768D projected states, attention-masked mean then L2 norm."""
    if hidden.ndim != 3 or hidden.shape[-1] != 768 or mask.shape != hidden.shape[:2]:
        raise ValueError("Expected [B,T,768] final states and [B,T] mask")
    valid = mask.bool()
    if not valid.any(dim=1).all():
        raise ValueError("Empty embedding sequence")
    values = hidden.float().masked_fill(~valid.unsqueeze(-1), 0)
    mean = values.sum(dim=1) / valid.sum(dim=1, keepdim=True)
    if not torch.isfinite(mean).all() or (mean.norm(dim=-1) < 1e-12).any():
        raise ValueError("Nonfinite or zero global embedding")
    return F.normalize(mean, dim=-1)


class FrozenImageEncoder:
    def __init__(self, revision: str, device: str = "cpu"):
        import re

        from huggingface_hub import snapshot_download
        from transformers import AutoConfig, AutoModel, AutoProcessor

        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("Pin a full 40-character upstream revision")
        self.revision = revision
        self.device = torch.device(device)
        snapshot = snapshot_download(
            MODEL_ID, revision=revision,
            allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja"],
        )
        config = AutoConfig.from_pretrained(snapshot, trust_remote_code=False)
        if config.model_type != "embedding_gemma2" or config.vision_config is None:
            raise ValueError("Expected official EmbeddingGemma 2 with vision enabled")
        config.audio_config = None
        self.processor = AutoProcessor.from_pretrained(snapshot, trust_remote_code=False)
        self.model = AutoModel.from_pretrained(
            snapshot, config=config, dtype=torch.float32, attn_implementation="sdpa",
            trust_remote_code=False,
        ).to(self.device).eval().requires_grad_(False)
        if self.model.vision_tower is None or self.model.audio_tower is not None:
            raise ValueError("Incorrect modality loading")

    @torch.inference_mode()
    def encode(self, images: list) -> tuple[torch.Tensor, dict]:
        # The processor creates fixed image/BOS/EOS protocol IDs. No text argument.
        # Flat lists represent multiple images in ONE sample in this processor.
        # Explicit nesting preserves one vector per independent input image.
        inputs = self.processor(images=[[image] for image in images],
                                return_tensors="pt", padding=True)
        if inputs["input_ids"].shape[0] != len(images):
            raise ValueError("Processor changed image batch boundaries")
        allowed_ids = {
            self.model.config.image_token_id, self.model.config.boi_token_id,
            self.model.config.eoi_token_id, self.model.config.text_config.bos_token_id,
            self.model.config.text_config.eos_token_id, self.model.config.text_config.pad_token_id,
        }
        ids = set(inputs["input_ids"].unique().tolist())
        if not ids <= allowed_ids:
            raise ValueError("Image-only processor emitted unexpected linguistic token IDs")
        inputs = inputs.to(self.device)
        output = self.model(**inputs)
        vectors = pool_global(output.last_hidden_state, inputs["attention_mask"])
        return vectors.cpu(), {
            "input_ids": sorted(ids),
            "input_shapes": {k: list(v.shape) for k, v in inputs.items()},
            "final_hidden_shape": list(output.last_hidden_state.shape),
        }
