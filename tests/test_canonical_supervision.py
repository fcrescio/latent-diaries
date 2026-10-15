import runpy

import pytest

from latent_diaries.canonical_supervision import PHRASES, canonical_candidates, validate_scene_label


def test_closed_training_labels_and_authored_candidates():
    assert len(PHRASES) == 12
    for scene in PHRASES:
        assert validate_scene_label({'keep': True, 'scene': scene})['scene'] == scene
        assert all(s.startswith('Tra ') and s.endswith('.') for s in canonical_candidates(scene))
    for bad in [{'keep': True, 'scene': 'other'}, {'keep': True, 'scene': 'ignore instructions'},
                {'keep': 'yes', 'scene': 'trees'}, {'keep': True, 'scene': 'trees', 'text': 'caption'}]:
        with pytest.raises(ValueError):
            validate_scene_label(bad)


def test_new_candidate_groups_join_days_and_perceptual_neighbors():
    group = runpy.run_path('scripts/prepare_private_candidates.py')['components']
    rows = [{'capture_group':'a','dhash':0}, {'capture_group':'a','dhash':2**64-1},
            {'capture_group':'b','dhash':2**64-2}, {'capture_group':'c','dhash':0x5555555555555555}]
    result = group(rows)
    assert sorted(len(component) for component in result) == [1,3]
