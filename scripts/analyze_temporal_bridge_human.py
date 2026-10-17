"""Seal aggregate TB human outcome and verify generated/teacher waveform parity."""

import argparse
import json
from pathlib import Path

from latent_diaries.feature_provenance import checksum
from latent_diaries.temporal_review import summarize_temporal_ratings


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["ratings", "mapping", "teachers", "corpus", "output"]:
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("Fresh outcome output required")
    ratings = json.loads(args.ratings.read_text())
    mapping = json.loads(args.mapping.read_text())["mapping"]
    teachers = json.loads(args.teachers.read_text())
    corpus = json.loads(args.corpus.read_text())
    if teachers["corpus_sha256"] != checksum(args.corpus):
        raise ValueError("Teacher/corpus pairing changed")
    by_clip = {r["clip_id"]: r for r in mapping}
    by_story = {r["id"]: r for r in corpus["rows"]}
    by_teacher = {r["id"]: r for r in teachers["records"]}
    if len(mapping) != 12 or len(by_clip) != 12 or {r["clip_id"] for r in ratings} != set(by_clip):
        raise ValueError("Actual ratings must cover the exact listening cohort")
    joined = []
    for rating in ratings:
        case = by_clip[rating["clip_id"]]
        teacher = by_teacher[case["history_sequence_id"] + "-f2"]
        student = case["student"]
        student_sha = checksum(student["audio"])
        teacher_sha = checksum(teacher["audio"])
        if student_sha != student["audio_sha256"] or teacher_sha != teacher["audio_sha256"]:
            raise ValueError("Generated or teacher audio changed")
        joined.append(
            {
                **rating,
                "mode": case["mode"],
                "split": case["split"],
                "kind": case["kind"],
                "image_id": by_story[case["sequence_id"]]["image_id"],
                "frames": student["frames"],
                "teacher_waveform_exact": student_sha == teacher_sha,
            }
        )
    summary = summarize_temporal_ratings(joined)
    validation_groups = [g for g in summary["groups"] if g["split"] == "validation"]
    validation_clips = sum(g["rated_clips"] for g in validation_groups)
    validation_bad = sum(g["explicit_gibberish_notes"] for g in validation_groups)
    reproduced_train = sum(
        g["teacher_waveform_byte_exact"] for g in summary["groups"] if g["split"] == "train"
    )
    report = {
        "scope": "Human assessment of the SECOND fragment; teacher first fragments are not student successes",
        "source_sha256": checksum(__file__),
        "analysis_source_sha256": checksum("src/latent_diaries/temporal_review.py"),
        "private_input_sha256": {
            name: checksum(getattr(args, name)) for name in ["ratings", "mapping", "teachers", "corpus"]
        },
        **summary,
        "training_teacher_waveform_reproductions": reproduced_train,
        "validation_explicit_gibberish_notes": validation_bad,
        "validation_reviewed_presentations": validation_clips,
        "decision": (
            "Stop this temporal regression recipe; retain encoder/corpus evidence separately."
            if validation_bad == validation_clips
            else "No automatic promotion; review the aggregate outcome."
        ),
        "new_training_performed": False,
        "m2_accepted": False,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "rated_clips": report["rated_clips"],
                "explicit_gibberish_notes": report["explicit_gibberish_notes"],
            }
        )
    )


if __name__ == "__main__":
    main()
