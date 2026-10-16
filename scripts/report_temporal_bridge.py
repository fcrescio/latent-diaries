"""Publish whitelisted bridge aggregates, never private narratives or ASR text."""

import argparse
import json
from pathlib import Path

from audit_audio_feedback_transcription import edit_distance, words

from latent_diaries.feature_provenance import checksum


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    root = args.root
    if args.output.exists():
        raise ValueError("Fresh aggregate report required")
    read = lambda name: json.loads((root / name).read_text())
    corpus = read("corpus-v2/corpus.json")
    stories = {r["id"]: r for r in corpus["rows"]}
    mapping = {r["id"]: r for r in read("bridge-evaluation-mapping.json")}
    asr = read("bridge-asr-small.json")
    if asr["audio_uploaded"]:
        raise ValueError("Local auxiliary diagnostic required")
    transcripts = {r["audio"]: r["auxiliary_transcription"] for r in asr["results"]}
    models = []
    caps = []
    lexical = []
    for mode in ["t0", "t1"]:
        config = read(f"bridge-{mode}/config.json")
        final = json.loads((root / f"bridge-{mode}/loss.jsonl").read_text().splitlines()[-1])
        models.append(
            {
                "mode": mode,
                **{
                    k: config[k]
                    for k in [
                        "feedback",
                        "completed_steps",
                        "schedule_steps",
                        "trainable_parameters",
                        "maximum_tokens",
                        "seed",
                        "reader_sha256",
                        "peak_allocated_vram_bytes",
                    ]
                },
                "final_training_masked_prefix_mse": final["prefix_mse"],
                "final_training_length_mse": final["length_mse"],
                "strict_resume_exercised": "--resume" in read(f"train-{mode}-2000.command.json")["argv"],
            }
        )
        results = read(f"bridge-evaluation-{mode}/results.json")
        if results["text_used"] or results["teacher_targets_read"] or results["seed"] != 42:
            raise ValueError("Strict inference protocol differs")
        for row in results["records"]:
            context = mapping[row["id"]]
            caps.append(
                {
                    "mode": mode,
                    "split": context["split"],
                    "kind": context["kind"],
                    "capped": row["frames"] >= 240,
                    "empty": row["audio"] is None,
                }
            )
            if context["kind"] not in {"matched", "same-image"} or row["audio"] is None:
                continue
            # Reference reproduction is an auxiliary bridge diagnostic; new
            # artistic continuations need not reproduce any teacher sentence.
            expected = words(stories[context["history_sequence_id"]]["fragments"][1])
            actual = words(transcripts[row["audio"]])
            lexical.append(
                {
                    "mode": mode,
                    "split": context["split"],
                    "kind": context["kind"],
                    "reference_words": len(expected),
                    "edits": edit_distance(expected, actual),
                }
            )

    def aggregate(rows, keys, metric):
        groups = sorted({tuple(r[k] for k in keys) for r in rows})
        return [
            {
                **dict(zip(keys, values, strict=True)),
                **metric([r for r in rows if tuple(r[k] for k in keys) == values]),
            }
            for values in groups
        ]

    cap_summary = aggregate(
        caps,
        ["mode", "split", "kind"],
        lambda rows: {
            "cases": len(rows),
            "capped": sum(r["capped"] for r in rows),
            "empty": sum(r["empty"] for r in rows),
        },
    )
    lexical_summary = aggregate(
        lexical,
        ["mode", "split", "kind"],
        lambda rows: {
            "cases": len(rows),
            "micro_reference_word_edit_ratio": sum(r["edits"] for r in rows)
            / sum(r["reference_words"] for r in rows),
        },
    )
    acoustic = read("bridge-acoustic-audit.json")
    acoustic_summary = [
        {
            k: r[k]
            for k in [
                "feedback",
                "split",
                "student",
                "oracle",
                "reader_gradient_norm",
                "backbone_gradients_absent",
            ]
        }
        for r in acoustic["records"]
    ]
    files = [
        "human-validation.json",
        "bridge-plan.json",
        "bridge-targets/manifest.json",
        "bridge-parity-probe/report.json",
        "bridge-decode-amendment.json",
        "bridge-numeric-diagnostics.json",
        "bridge-acoustic-audit.json",
        "bridge-asr-small.json",
        "bridge-listening/private-mapping.json",
    ]
    report = {
        "scope": "Bounded teacher-history continuous-prefix diagnostic, not an autonomous narrator or M2 success",
        "source_sha256": checksum(__file__),
        "private_evidence_sha256": {name: checksum(root / name) for name in files},
        "training_sequences": 16,
        "diagnostic_validation_sequences": 8,
        "training_images": 8,
        "validation_images": 4,
        "all_reserved96_images_untouched": True,
        "models": models,
        "numeric_diagnostics": read("bridge-numeric-diagnostics.json")["aggregate"],
        "contradictory_target_floors": read("bridge-numeric-diagnostics.json")[
            "contradictory_target_floors"
        ],
        "teacher_forced_acoustic_diagnostics": acoustic_summary,
        "acoustic_audit_peak_vram_bytes": acoustic["peak_vram_bytes"],
        "generation": cap_summary,
        "auxiliary_reference_reconstruction": lexical_summary,
        "asr_is_human_intelligibility_or_art_score": False,
        "decoding_parity": [
            {k: r[k] for k in ["mode", "prefill_exact", "codes_exact", "frames", "reached_cap"]}
            for r in read("bridge-parity-probe/report.json")["records"]
        ],
        "blinded_teacher_history_student_continuation_pairs": 12,
        "review_labels": "TB-01 through TB-12",
        "first_fragment_is_teacher": True,
        "second_fragment_is_textless_student": True,
        "human_student_quality_review_pending": True,
        "generated_first_fragment_rollout_exercised": False,
        "acoustic_optimization_exercised": False,
        "artistic_improvement_verified": False,
        "m2_accepted": False,
        "paid_jobs": 0,
        "private_uploads": 0,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"models": len(models), "generated_controls": len(caps), "review_pairs": 12}))


if __name__ == "__main__":
    main()
