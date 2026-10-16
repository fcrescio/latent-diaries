import copy

import pytest

from latent_diaries.temporal_corpus import select_pilot, validate_story


def test_selection_is_capture_disjoint_and_never_promotes_development_images():
    rows = [{'id': str(i), 'split': 'train', 'scene_group': str(i)} for i in range(30)]
    rows.append({'id': 'excluded', 'split': 'heldout', 'scene_group': 'excluded'})
    design = {r['id']: {**r, 'scene': str(i % 10), 'path': 'invented-path'} for i, r in enumerate(rows)}
    selected = select_pilot(rows, design)
    assert selected == select_pilot(list(reversed(rows)), design)
    assert len(selected) == len({r['scene_group'] for r in selected}) == 12
    assert len({r['scene'] for r in selected}) == 10
    assert sum(r['split'] == 'validation' for r in selected) == 4
    assert all(r['source_split'] == 'train' and r['id'] != 'excluded' for r in selected)


def test_structural_rejection_does_not_claim_semantic_validation():
    with pytest.raises(ValueError):
        validate_story({'keep': False, 'visible_details': [], 'branches': []})
    with pytest.raises(ValueError):
        validate_story({'keep': True, 'visible_details': ['generic', 'invented'], 'branches': []})
    with pytest.raises(ValueError):
        validate_story({'keep': True, 'visible_details': ['generic', 'invented'], 'branches': ['A', 'B']})


def test_valid_branch_structure_and_copied_continuation_rejection():
    fragments = [
        ['Un oggetto opaco occupava il centro del tavolo di prova.',
         'Per osservarne il bordo ho cambiato posizione senza spostare altro.'],
        ['Vicino al margine notavo una linea chiara sul fondo scuro.',
         "L'ho seguita con un dito finché spariva dietro la forma."]]
    value = {'keep': True, 'visible_details': ['generic object', 'invented line'],
             'branches': [{'key': k, 'focus_it': 'Synthetic test only', 'fragments': f,
                           'progression_it': 'Synthetic test only', 'a2_requires_a1_it': 'Synthetic test only'}
                          for k, f in zip(['A', 'B'], fragments, strict=True)]}
    assert validate_story(value) is value
    repeated = copy.deepcopy(value)
    repeated['branches'][0]['fragments'][1] = repeated['branches'][0]['fragments'][0]
    with pytest.raises(ValueError, match='Repeated'):
        validate_story(repeated)
