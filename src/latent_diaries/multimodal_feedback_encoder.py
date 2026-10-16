"""Frozen media-only feedback encoder; historical FrozenImageEncoder is untouched."""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample_poly

from .vision import MODEL_ID, pool_global

SAMPLE_RATE = 16000


def prepare_audio(audio, *, maximum_seconds=30.0):
    """Decode locally, downmix [frames,channels], resample; never truncate a story."""
    if isinstance(audio, (str, Path)):
        import soundfile as sf
        values, sample_rate = sf.read(audio, dtype='float32', always_2d=False)
    elif isinstance(audio, tuple) and len(audio) == 2:
        values, sample_rate = audio
        values = np.asarray(values)
    else:
        raise ValueError('Local audio path or (waveform,sample_rate) required')
    if type(sample_rate) is not int or not 1 <= sample_rate <= 384000:
        raise ValueError('Valid integer waveform sample rate required')
    if values.ndim not in {1, 2} or (values.ndim == 2 and not 1 <= values.shape[1] <= 8):
        raise ValueError('Expected mono or [frames,channels] waveform')
    if not np.isfinite(values).all() or len(values) == 0:
        raise ValueError('Nonempty finite audio required')
    seconds = len(values) / sample_rate
    if not 0.1 <= seconds <= maximum_seconds:
        raise ValueError('Audio duration outside explicit feedback bounds; no silent truncation')
    channels = 1 if values.ndim == 1 else values.shape[1]
    mono = values.astype(np.float32).mean(-1) if values.ndim == 2 else values.astype(np.float32)
    if float(np.sqrt(np.mean(mono.astype(np.float64)**2))) < 1e-7:
        raise ValueError('Silent or empty history is not absent-history conditioning')
    if sample_rate != SAMPLE_RATE:
        factor = math.gcd(sample_rate, SAMPLE_RATE)
        mono = resample_poly(mono, SAMPLE_RATE // factor, sample_rate // factor).astype(np.float32)
        mono = mono[:round(seconds * SAMPLE_RATE)]
    if not np.isfinite(mono).all() or len(mono) == 0:
        raise ValueError('Invalid resampled waveform')
    return np.ascontiguousarray(mono), {'original_sample_rate': sample_rate, 'original_channels': channels,
        'sample_rate': SAMPLE_RATE, 'samples': len(mono), 'seconds': len(mono)/SAMPLE_RATE}


class FrozenFeedbackEncoder:
    """Image, audio and native joint representations; no text-bearing public API."""
    def __init__(self, revision, device='cpu', dtype='float32', maximum_seconds=30.0):
        from huggingface_hub import snapshot_download
        from transformers import AutoConfig, AutoModel, AutoProcessor

        if not re.fullmatch(r'[0-9a-f]{40}', revision):
            raise ValueError('Pin a full upstream revision')
        if dtype not in {'float32', 'bfloat16'}:
            raise ValueError('Only float32 or supported bfloat16; never float16')
        self.device = torch.device(device)
        if dtype == 'bfloat16' and (self.device.type != 'cuda' or not torch.cuda.is_bf16_supported()):
            raise ValueError('This path requires native CUDA bfloat16 support')
        if not 0.1 <= maximum_seconds <= 30:
            raise ValueError('Feedback audio bound must be explicit and at most30 seconds')
        self.revision, self.dtype, self.maximum_seconds = revision, dtype, maximum_seconds
        snapshot = snapshot_download(MODEL_ID, revision=revision, local_files_only=True,
            allow_patterns=['*.json', '*.safetensors', '*.model', '*.txt', '*.jinja'])
        config = AutoConfig.from_pretrained(snapshot, trust_remote_code=False)
        if config.model_type != 'embedding_gemma2' or config.vision_config is None or config.audio_config is None:
            raise ValueError('Pinned full EmbeddingGemma2 vision/audio configuration required')
        self.processor = AutoProcessor.from_pretrained(snapshot, trust_remote_code=False)
        self.model = AutoModel.from_pretrained(snapshot, config=config, dtype=getattr(torch, dtype),
            attn_implementation='sdpa', trust_remote_code=False).to(self.device).eval().requires_grad_(False)
        if self.model.vision_tower is None or self.model.audio_tower is None:
            raise ValueError('Both frozen media towers must be loaded')
        self.protocol_ids = {getattr(config, key) for key in
            ['image_token_id', 'boi_token_id', 'eoi_token_id', 'audio_token_id', 'boa_token_id', 'eoa_token_index']}
        self.protocol_ids.update(getattr(config.text_config, k) for k in ['bos_token_id', 'eos_token_id', 'pad_token_id'])
        # The pinned processor joins fixed modality placeholders with whitespace.
        self.protocol_ids.update(self.processor.tokenizer.encode(' ', add_special_tokens=False))

    @torch.inference_mode()
    def _encode(self, *, images=None, audio=None):
        if images is None and audio is None:
            raise ValueError('At least one media modality required')
        count = len(images) if images is not None else len(audio)
        if count < 1 or (images is not None and audio is not None and len(images) != len(audio)):
            raise ValueError('Nonempty aligned media batches required')
        kwargs = {'return_tensors': 'pt', 'padding': True}
        records = []
        if images is not None:
            kwargs['images'] = [[image] for image in images]
        if audio is not None:
            prepared = [prepare_audio(a, maximum_seconds=self.maximum_seconds) for a in audio]
            kwargs['audio'] = [a for a, _ in prepared]
            records = [r for _, r in prepared]
            kwargs['audio_kwargs'] = {'sampling_rate': SAMPLE_RATE, 'truncation': False, 'max_length': 480000}
        # No text parameter: pinned upstream derives only fixed modality markers.
        inputs = self.processor(**kwargs)
        if inputs['input_ids'].shape[0] != count:
            raise ValueError('Processor changed independent sample boundaries')
        ids = set(inputs['input_ids'].unique().tolist())
        if not ids <= self.protocol_ids:
            raise ValueError('Media processor emitted variable/non-protocol text IDs')
        inputs = inputs.to(self.device)
        if self.dtype == 'bfloat16':
            for key, value in inputs.items():
                if value.is_floating_point():
                    inputs[key] = value.to(torch.bfloat16)
        output = self.model(**inputs)
        vector = pool_global(output.last_hidden_state, inputs['attention_mask']).cpu()
        return vector, {'modality': 'joint' if images is not None and audio is not None else ('image' if images is not None else 'audio'),
            'input_ids': sorted(ids), 'input_shapes': {k: list(v.shape) for k, v in inputs.items()},
            'final_hidden_shape': list(output.last_hidden_state.shape), 'audio_conversion': records,
            'variable_text_used': False, 'dtype': self.dtype, 'revision': self.revision}

    def encode_image(self, images):
        return self._encode(images=images)

    def encode_audio(self, audio):
        return self._encode(audio=audio)

    def encode_image_audio(self, images, audio):
        return self._encode(images=images, audio=audio)
