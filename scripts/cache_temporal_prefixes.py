"""Training-only pinned Qwen projected targets for the validated continuation pilot."""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from latent_diaries.inversion import digest, load_backbone, render


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["corpus", "validation", "teachers", "media-cache", "output-dir"]:
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    if not Path("/.dockerenv").exists() or args.output_dir.exists():
        raise ValueError("Managed worker and fresh target directory required")
    corpus = json.loads(args.corpus.read_text())
    validation = json.loads(args.validation.read_text())
    teachers = json.loads(args.teachers.read_text())
    media = json.loads((args.media_cache / "manifest.json").read_text())
    if (
        not validation["human_subset_validated"]
        or validation["corpus_sha256"] != digest(args.corpus)
        or validation["teacher_manifest_sha256"] != digest(args.teachers)
        or teachers["corpus_sha256"] != digest(args.corpus)
    ):
        raise ValueError("Human subset validation must bind this exact corpus and teacher manifest")
    if media["features_sha256"] != digest(args.media_cache / "features.safetensors"):
        raise ValueError("Media cache changed")
    from safetensors.torch import load_file

    features = load_file(str(args.media_cache / "features.safetensors"))
    images = dict(zip(media["image_ids"], features["image"], strict=True))
    audio = dict(zip(media["audio_ids"], features["audio"], strict=True))
    teacher_by_id = {r["id"]: r for r in teachers["records"]}
    info = json.loads(Path("artifacts/m0_italian.json").read_text())
    wrapper, builder, speaker = load_backbone(info)
    args.output_dir.mkdir(mode=0o700)
    records = []
    with torch.no_grad():
        for row in corpus["rows"]:
            key = row["id"]
            teacher = teacher_by_id[key + "-f2"]
            if digest(Path(teacher["codes"])) != teacher["codes_sha256"]:
                raise ValueError("Gold codec target changed")
            ids = wrapper._tokenize_texts([wrapper._build_assistant_text(row["fragments"][1])])[0][
                :, 3:-5
            ]
            prefix = wrapper.model.talker.text_projection(
                wrapper.model.talker.get_text_embeddings()(ids)
            )
            if prefix.shape[0] != 1 or prefix.shape[2] != 1024 or not 1 <= prefix.shape[1] <= 96:
                raise ValueError("Pinned projected prefix shape changed")
            file = args.output_dir / (key + ".safetensors")
            save_file({"soft_tokens": prefix.cpu().contiguous()}, str(file))
            records.append(
                {
                    "id": key,
                    "split": row["split"],
                    "scene_group": row["scene_group"],
                    "image_id": row["image_id"],
                    "image": images[row["image_id"]].tolist(),
                    "audio": audio[key + "-f1"].tolist(),
                    "prefix": str(file),
                    "prefix_sha256": digest(file),
                    "length": prefix.shape[1],
                    "codes": teacher["codes"],
                    "codes_sha256": teacher["codes_sha256"],
                }
            )
        # Fixed first sequence from each split, selected by identity before listening.
        controls = []
        for split in ["train", "validation"]:
            row = min((r for r in records if r["split"] == split), key=lambda r: r["id"])
            prefix = load_file(row["prefix"])["soft_tokens"].to(speaker.device)
            controls.append(
                {
                    "id": row["id"],
                    "split": split,
                    **render(
                        builder,
                        wrapper,
                        prefix,
                        speaker,
                        args.output_dir / (split + "-oracle.wav"),
                        max_frames=240,
                        seed=42,
                        codec0_greedy=True,
                    ),
                }
            )
    manifest = {
        "scope": "Offline training targets; exact textual projection oracle is not textless inference",
        "corpus_sha256": digest(args.corpus),
        "validation_sha256": digest(args.validation),
        "teacher_manifest_sha256": digest(args.teachers),
        "media_manifest_sha256": digest(args.media_cache / "manifest.json"),
        "model_id": info["model_id"],
        "model_revision": info["model_revision"],
        "role_ids": list(builder.role_ids),
        "source_sha256": digest(Path(__file__)),
        "lock_sha256": digest(Path("uv.lock")),
        "records": records,
        "oracle_controls": controls,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"cached_continuations": len(records), "oracle_controls": len(controls)}))


if __name__ == "__main__":
    main()
