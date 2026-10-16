"""Offline bridge errors, contradictory-target floor and image/history interventions."""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from latent_diaries.feature_provenance import checksum
from latent_diaries.temporal_reader import TemporalReader


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--targets", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, action="append", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if not Path("/.dockerenv").exists() or args.output.exists():
        raise ValueError("Managed worker and fresh offline diagnostic required")
    torch.set_num_threads(4)
    data = json.loads(args.targets.read_text())
    rows = data["records"]
    targets = {}
    for row in rows:
        if checksum(row["prefix"]) != row["prefix_sha256"]:
            raise ValueError("Evaluation target changed")
        targets[row["id"]] = load_file(row["prefix"])["soft_tokens"]
    results = []
    floors = []
    with torch.no_grad():
        for run in args.run_dir:
            info = json.loads((run / "config.json").read_text())
            if (
                info["targets_sha256"] != checksum(args.targets)
                or checksum(run / "reader.safetensors") != info["reader_sha256"]
            ):
                raise ValueError("Checkpoint/target provenance differs")
            weights = load_file(str(run / "reader.safetensors"))
            reader = TemporalReader(
                info["maximum_tokens"],
                weights["mean"],
                weights["scale"],
                feedback=info["feedback"],
                width=info["width"],
            )
            reader.load_state_dict(weights)
            reader.eval()
            maximum = reader.maximum_tokens
            if not info["feedback"]:
                numerator = denominator = length_floor = 0.0
                training = [r for r in rows if r["split"] == "train"]
                for key in sorted({r["image_id"] for r in training}):
                    group = [r for r in training if r["image_id"] == key]
                    if len(group) != 2 or group[0]["image"] != group[1]["image"]:
                        raise ValueError("Exact same-image branch pairing required")
                    a, b = [targets[r["id"]][0] for r in group]
                    common = min(len(a), len(b))
                    # Two residuals around their shared least-squares optimum.
                    numerator += float((((a[:common] - b[:common]) / reader.scale).square() / 2).sum())
                    denominator += (len(a) + len(b)) * 1024
                    length_floor += ((len(a) - len(b)) / maximum) ** 2 / 2
                floors.append(
                    {
                        "image_only_masked_prefix_floor": numerator / denominator,
                        "image_only_length_floor": length_floor / len(training),
                    }
                )
            for row in rows:
                other = next(
                    r for r in rows if r["image_id"] == row["image_id"] and r["id"] != row["id"]
                )
                cross = next(
                    r for r in rows if r["split"] == row["split"] and r["image_id"] != row["image_id"]
                )
                image = torch.tensor([row["image"]])
                audio = torch.tensor([row["audio"]])
                present = torch.ones(1, dtype=torch.bool)
                matched, _, lengths, _ = reader(image, audio, present)
                swapped = reader(image, torch.tensor([other["audio"]]), present)[0]
                changed_image = reader(torch.tensor([cross["image"]]), audio, present)[0]
                absent = reader(image, audio, ~present)[0]
                target = targets[row["id"]]
                other_target = targets[other["id"]]
                if target.shape[1] > maximum or other_target.shape[1] > maximum:
                    raise ValueError(
                        "Gold evaluation prefix exceeds trained capacity; report rather than truncate"
                    )

                def error(predicted, gold, _scale=reader.scale):
                    return float(((predicted[:, : gold.shape[1]] - gold) / _scale).square().mean())

                results.append(
                    {
                        "feedback": info["feedback"],
                        "id": row["id"],
                        "split": row["split"],
                        "matched_prefix_mse": error(matched, target),
                        "swapped_to_other_branch_mse": error(swapped, other_target),
                        "length_absolute_error": abs(int(lengths[0]) - target.shape[1]),
                        "same_image_history_prefix_delta": error(swapped, matched),
                        "image_intervention_prefix_delta": error(changed_image, matched),
                        "absent_history_prefix_delta": error(absent, matched),
                    }
                )
    aggregate = []
    for feedback in [False, True]:
        for split in ["train", "validation"]:
            selected = [r for r in results if r["feedback"] == feedback and r["split"] == split]
            aggregate.append(
                {
                    "feedback": feedback,
                    "split": split,
                    "cases": len(selected),
                    **{
                        k: sum(r[k] for r in selected) / len(selected)
                        for k in selected[0]
                        if k.endswith(("_mse", "_error", "_delta"))
                    },
                }
            )
    args.output.write_text(
        json.dumps(
            {
                "scope": "Numeric offline diagnostics only, not artistic or human quality scores",
                "source_sha256": checksum(__file__),
                "targets_sha256": checksum(args.targets),
                "contradictory_target_floors": floors,
                "aggregate": aggregate,
                "records": results,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"aggregate": aggregate, "floors": floors}), flush=True)


if __name__ == "__main__":
    main()
