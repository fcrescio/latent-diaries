import inspect
from types import SimpleNamespace

import pytest

torch = pytest.importorskip('torch')
import numpy as np

from latent_diaries.multimodal_feedback_encoder import FrozenFeedbackEncoder, prepare_audio


def test_resampling_mono_and_silent_invalid_history_bounds():
    t = np.arange(44100) / 44100
    signal = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    mono, info = prepare_audio((np.stack([signal, signal/2], -1), 44100))
    assert mono.shape == (16000,) and mono.dtype == np.float32
    assert info['original_channels'] == 2 and info['sample_rate'] == 16000
    spectrum = np.abs(np.fft.rfft(mono))
    assert int(spectrum.argmax()) == 440
    for value in [(np.zeros(16000), 16000), (np.ones(1), 16000),
                  (np.ones(16000*31), 16000), (np.full(16000, np.nan), 16000)]:
        with pytest.raises(ValueError):
            prepare_audio(value)


def test_media_api_never_accepts_linguistic_conditioning():
    for method in [FrozenFeedbackEncoder.encode_image, FrozenFeedbackEncoder.encode_audio,
                   FrozenFeedbackEncoder.encode_image_audio]:
        assert not {'text', 'transcript', 'caption'} & set(inspect.signature(method).parameters)
    instance = FrozenFeedbackEncoder.__new__(FrozenFeedbackEncoder)
    with pytest.raises(TypeError):
        instance.encode_audio([], text='hidden words')


def test_processor_boundary_and_modal_separation_without_model_download():
    class Batch(dict):
        def to(self, device):
            return self
    class Processor:
        def __init__(self):
            self.calls = []
            self.token = 1
        def __call__(self, **kwargs):
            self.calls.append(kwargs)
            assert 'text' not in kwargs
            n = len(kwargs.get('images', kwargs.get('audio')))
            return Batch(input_ids=torch.full((n, 2), self.token), attention_mask=torch.ones(n, 2))
    class Model:
        def __call__(self, **kwargs):
            n = len(kwargs['input_ids'])
            return SimpleNamespace(last_hidden_state=torch.ones(n, 2, 768))
    encoder = FrozenFeedbackEncoder.__new__(FrozenFeedbackEncoder)
    encoder.processor, encoder.model = Processor(), Model()
    encoder.device, encoder.dtype = torch.device('cpu'), 'float32'
    encoder.revision = 'a' * 40
    encoder.maximum_seconds, encoder.protocol_ids = 30, {1}
    waveform = (np.ones(16000, dtype=np.float32)*.1, 16000)
    for vector, meta in [encoder.encode_image(['image']), encoder.encode_audio([waveform]),
                         encoder.encode_image_audio(['image'], [waveform])]:
        assert vector.shape == (1, 768) and torch.isfinite(vector).all()
        torch.testing.assert_close(vector.norm(dim=-1), torch.ones(1))
        assert meta['variable_text_used'] is False
    assert ['images' in c for c in encoder.processor.calls] == [True, False, True]
    assert ['audio' in c for c in encoder.processor.calls] == [False, True, True]
    encoder.processor.token = 999
    with pytest.raises(ValueError, match='non-protocol'):
        encoder.encode_audio([waveform])
    with pytest.raises(ValueError, match='aligned'):
        encoder.encode_image_audio(['image'], [waveform, waveform])
