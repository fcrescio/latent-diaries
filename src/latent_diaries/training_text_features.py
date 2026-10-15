"""Training/diagnostic text embeddings only; never import in image inference."""
import re

import torch

from .vision import MODEL_ID, pool_global


class FrozenTrainingTextEncoder:
    def __init__(self, revision, device='cpu'):
        from huggingface_hub import snapshot_download
        from transformers import AutoConfig, AutoModel, AutoProcessor

        if not re.fullmatch(r'[0-9a-f]{40}', revision):
            raise ValueError('Pin the full upstream revision')
        snapshot = snapshot_download(MODEL_ID, revision=revision, local_files_only=True,
                                     allow_patterns=['*.json', '*.safetensors', '*.model', '*.txt', '*.jinja'])
        config = AutoConfig.from_pretrained(snapshot, trust_remote_code=False)
        config.audio_config = config.vision_config = None
        self.processor = AutoProcessor.from_pretrained(snapshot, trust_remote_code=False)
        self.model = AutoModel.from_pretrained(snapshot, config=config, dtype=torch.float32,
                                              attn_implementation='sdpa', trust_remote_code=False)
        self.device = torch.device(device)
        self.model.to(self.device).eval().requires_grad_(False)
        if self.model.vision_tower is not None or self.model.audio_tower is not None:
            raise ValueError('Training-text encoder must load neither media tower')

    @torch.inference_mode()
    def encode(self, texts):
        if not texts or any(not isinstance(t, str) or not t.strip() for t in texts):
            raise ValueError('Nonempty training texts required')
        inputs = self.processor(text=['title: none | text: ' + t for t in texts],
                                return_tensors='pt', padding=True).to(self.device)
        output = self.model(**inputs)
        return pool_global(output.last_hidden_state, inputs['attention_mask']).cpu()
