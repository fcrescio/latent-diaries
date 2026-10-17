import pytest

from latent_diaries.temporal_review import summarize_temporal_ratings


def fixture_rows():
    rows = []
    for split in ["train", "validation"]:
        for mode, kind in [("t0", "matched"), ("t1", "matched"), ("t1", "same-image")]:
            for image in [0, 1]:
                rows.append(
                    {
                        "clip_id": f"{split}-{mode}-{kind}-{image}",
                        "reviewer": "private reviewer",
                        "image_id": f"{split}-{image}",
                        "mode": mode,
                        "kind": kind,
                        "split": split,
                        "grounding": 1,
                        "voice": 1,
                        "evocation": 1,
                        "surprise": 1,
                        "note": "private note",
                        "teacher_waveform_exact": False,
                        "frames": 80,
                    }
                )
    return rows


def test_complete_review_preserves_negative_evidence_without_publishing_notes():
    rows = fixture_rows()
    for row in rows:
        if row["mode"] == "t0" or row["split"] == "validation":
            row["note"] = "glibberish"
        else:
            row.update(note="", voice=3, teacher_waveform_exact=True)
    report = summarize_temporal_ratings(rows)
    assert report["explicit_gibberish_notes"] == 8
    assert sum(r["teacher_waveform_byte_exact"] for r in report["groups"]) == 4
    assert report["note_absence_is_not_approval"]
    assert "private reviewer" not in str(report) and "train-0" not in str(report)
    assert not report["population_failure_rate_estimated"]


def test_missing_or_duplicate_ratings_are_not_imputed():
    rows = fixture_rows()
    with pytest.raises(ValueError):
        summarize_temporal_ratings(rows[:-1])
    with pytest.raises(ValueError):
        summarize_temporal_ratings(rows[:-1] + [rows[0]])


def test_comparison_requires_the_same_images_in_all_conditions():
    rows = fixture_rows()
    rows[0]["image_id"] = "unpaired-image"
    with pytest.raises(ValueError):
        summarize_temporal_ratings(rows)
