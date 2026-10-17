import pytest

from latent_diaries.omnivoice_protocol import get_plan, validate_reuse


def test_frozen_twenty_case_grid_and_exact_smoke_subset():
    full = get_plan("full")
    smoke = get_plan("smoke")
    assert len(full) == len(set(full)) == 20 and len(smoke) == 4
    assert set(smoke) <= set(full)
    assert sum(mode == "positive" for mode, *_ in full) == 2
    for mode in ["voice-empty", "lang-empty", "mask-only"]:
        assert {(seed, duration) for m, seed, duration, _ in full if m == mode} == {
            (seed, duration) for seed in [41, 42, 43] for duration in [4.0, 6.0]
        }
    with pytest.raises(ValueError):
        get_plan("adaptive")


def test_changed_recipe_or_partial_resume_cannot_be_reused():
    record = {
        "run_provenance_sha256": "run",
        "mode": "voice-empty",
        "seed": 41,
        "requested_duration_seconds": 4.0,
        "wav_sha256": "wav",
        "codes_sha256": "codes",
        "technical_valid": True,
    }
    validate_reuse(record, "run", "voice-empty", 41, 4.0, "wav", "codes")
    for changed in ["new-harness", "different-dtype", "different-guidance"]:
        with pytest.raises(RuntimeError):
            validate_reuse(record, changed, "voice-empty", 41, 4.0, "wav", "codes")
    with pytest.raises(RuntimeError):
        validate_reuse(record, "run", "voice-empty", 42, 4.0, "wav", "codes")
    with pytest.raises(RuntimeError):
        validate_reuse(record, "run", "voice-empty", 41, 4.0, "changed-wave", "codes")
