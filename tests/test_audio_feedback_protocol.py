import importlib.util
from pathlib import Path

import pytest


def test_minimal_attribute_controls_preserve_punctuation_and_change_one_word():
    path = Path('scripts/prepare_audio_feedback_benchmark.py')
    spec = importlib.util.spec_from_file_location('feedback_benchmark', path)
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert module.replace_attribute('Oggetto opaco, sfondo uniforme.', 'opaco', 'lucido') == 'Oggetto lucido, sfondo uniforme.'
    assert module.replace_attribute('Forma grande, vicino al margine.', 'grande', 'piccola') == 'Forma piccola, vicino al margine.'
    for text in ['nero e nero', 'nerofumo']:
        with pytest.raises(ValueError):
            module.replace_attribute(text, 'nero', 'bianco')
