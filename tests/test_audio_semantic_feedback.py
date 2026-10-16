import pytest

np = pytest.importorskip('numpy')
from latent_diaries.audio_semantic_feedback import select_audio_candidates


def test_numeric_feedback_selection_depends_on_anchor_and_keeps_baseline_failures():
    image = np.zeros(768);image[0] = 1
    wrong = np.zeros(768);wrong[1] = 1
    audio = np.stack([wrong,image,-image,wrong])
    result = select_audio_candidates(image,audio,[True]*4,source_id='opaque',shuffled_image=wrong)
    assert result['indices']['image_audio_max'] == 1
    assert result['indices']['shuffled_image_scorer'] == 0
    assert result['indices']['zero_image_scorer'] == 0
    assert result == select_audio_candidates(image,audio,[True]*4,source_id='opaque',shuffled_image=wrong)
    failed = select_audio_candidates(image,audio,[False]*4,source_id='opaque')
    assert all(v is None for v in failed['indices'].values())
    assert not result['quality_verified']
    with pytest.raises(ValueError):
        select_audio_candidates(np.zeros(768),audio,[True]*4,source_id='opaque')
