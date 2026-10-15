"""Prepare fresh local scene groups before labelling; never print private paths."""
import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageOps


def perceptual_hash(image):
    small = image.convert('L').resize((9, 8))
    values = list(small.get_flattened_data())
    return sum((values[y*9+x] > values[y*9+x+1]) << (y*8+x)
               for y in range(8) for x in range(8))


def components(rows):
    parent = list(range(len(rows)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, row in enumerate(rows):
        for j, other in enumerate(rows[:i]):
            if row['capture_group'] == other['capture_group'] or (row['dhash'] ^ other['dhash']).bit_count() <= 6:
                parent[root(i)] = root(j)
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[root(i)].append(row)
    return list(groups.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--paths-file', type=Path, required=True)
    parser.add_argument('--exclude-inventory', type=Path, required=True)
    parser.add_argument('--exclude-design', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=768)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError('Use fresh candidate directory')
    old = json.loads(args.exclude_inventory.read_text())
    old_design = json.loads(args.exclude_design.read_text())['rows']
    dates = {r['capture'][:10] if r['capture'] else r['group'] for r in old}
    old_paths = {r['path'] for r in old}
    old_dhash = [r['dhash'] for r in old_design]
    fingerprints = {r['image_sha256'] for r in old_design}
    paths = args.paths_file.read_text().splitlines()
    random.Random(20261015).shuffle(paths)
    rows = []
    for value in paths:
        path = Path(value)
        if value in old_paths or not path.resolve().is_relative_to(args.root.resolve()):
            continue
        try:
            with Image.open(path) as original:
                if min(original.size) < 600:
                    continue
                exif = original.getexif()
                capture = exif.get(36867) or exif.get(306)
                group = hashlib.sha256(str(path.parent).encode()).hexdigest()[:12]
                capture_group = str(capture)[:10] if capture else group
                if capture_group in dates:
                    continue
                image = ImageOps.exif_transpose(original)
                dhash = perceptual_hash(image)
                if any((dhash ^ previous).bit_count() <= 6 for previous in old_dhash):
                    continue
                fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()
                if fingerprint in fingerprints:
                    continue
                fingerprints.add(fingerprint)
                size = list(original.size)
            rows.append({'id': f'q{len(rows):04d}', 'path': str(path), 'group': group,
                         'capture': str(capture) if capture else None, 'capture_group': capture_group,
                         'dhash': dhash, 'image_sha256': fingerprint, 'size': size})
            if len(rows) % 64 == 0:
                print(f'Prepared {len(rows)} fresh local candidates', flush=True)
        except (OSError, ValueError, Image.DecompressionBombError):
            continue
        if len(rows) >= args.limit:
            break
    grouped = components(rows)
    random.Random(20261016).shuffle(grouped)
    # Reserve complete groups before labels/features/model performance are seen.
    counts = {'train': 0, 'development': 0, 'fresh_test': 0}
    selected = []
    for group in grouped:
        if counts['fresh_test'] < 96:
            split = 'fresh_test'
        elif counts['development'] < 96:
            split = 'development'
        else:
            split = 'train'
        scene_group = hashlib.sha256('|'.join(sorted(r['id'] for r in group)).encode()).hexdigest()[:16]
        selected.extend(r | {'split': split, 'scene_group': scene_group} for r in group)
        counts[split] += len(group)
    if min(counts.values()) == 0:
        raise ValueError('Insufficient independent groups')
    args.output_dir.mkdir(parents=True, mode=0o700)
    output = args.output_dir / 'inventory.json'
    output.write_text(json.dumps(sorted(selected, key=lambda r:r['id']), indent=2)+'\n')
    output.chmod(0o600)
    print(json.dumps({'candidates': len(rows), 'groups': len(grouped), 'split_counts': counts}), flush=True)


if __name__ == '__main__':
    main()
