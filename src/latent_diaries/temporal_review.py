"""Aggregate actual continuation reviews without publishing identities or notes."""

from collections import defaultdict
from statistics import mean


def summarize_temporal_ratings(rows):
    if len(rows) != 12 or len({r["clip_id"] for r in rows}) != 12:
        raise ValueError("All twelve distinct prespecified ratings required")
    if len({r["reviewer"] for r in rows}) != 1:
        raise ValueError("One common reviewer required for this descriptive report")
    fields = ("grounding", "voice", "evocation", "surprise")
    grouped = defaultdict(list)
    for row in rows:
        if any(type(row[k]) is not int or not 1 <= row[k] <= 5 for k in fields):
            raise ValueError("Actual bounded ordinal scores required")
        if row["mode"] not in {"t0", "t1"} or row["split"] not in {"train", "validation"}:
            raise ValueError("Unexpected model/split")
        if row["kind"] not in {"matched", "same-image"} or (
            row["mode"] == "t0" and row["kind"] != "matched"
        ):
            raise ValueError("Unexpected listening condition")
        grouped[row["mode"], row["split"], row["kind"]].append(row)
    expected = {
        (mode, split, kind)
        for split in ["train", "validation"]
        for mode, kind in [("t0", "matched"), ("t1", "matched"), ("t1", "same-image")]
    }
    if set(grouped) != expected or any(len(group) != 2 for group in grouped.values()):
        raise ValueError("Complete two-image condition cells required")
    for split in ["train", "validation"]:
        cells = [
            {r["image_id"] for r in group} for (_, part, _), group in grouped.items() if part == split
        ]
        if any(len(cell) != 2 or cell != cells[0] for cell in cells):
            raise ValueError("Each model must evaluate the same two images per split")
    if {r["image_id"] for r in rows if r["split"] == "train"} & {
        r["image_id"] for r in rows if r["split"] == "validation"
    }:
        raise ValueError("Training/validation image overlap")

    def bad_note(row):
        note = row["note"].casefold()
        return "glibberish" in note or "gibberish" in note

    groups = []
    for (mode, split, kind), group in sorted(grouped.items()):
        groups.append(
            {
                "mode": mode,
                "split": split,
                "kind": kind,
                "rated_clips": len(group),
                "source_images": len({r["image_id"] for r in group}),
                "explicit_gibberish_notes": sum(bad_note(r) for r in group),
                "voice_at_one": sum(r["voice"] == 1 for r in group),
                "teacher_waveform_byte_exact": sum(r["teacher_waveform_exact"] for r in group),
                "duration_caps": sum(r["frames"] >= 240 for r in group),
                "mean_scores": {k: mean(r[k] for r in group) for k in fields},
            }
        )
    return {
        "rated_clips": len(rows),
        "source_images": len({r["image_id"] for r in rows}),
        "reviewers": 1,
        "explicit_gibberish_notes": sum(bad_note(r) for r in rows),
        "note_absence_is_not_approval": True,
        "voice_score_is_not_dedicated_intelligibility_scale": True,
        "groups": groups,
        "population_failure_rate_estimated": False,
        "independent_image_generalization_verified": False,
        "human_quality_improvement_accepted": False,
    }
