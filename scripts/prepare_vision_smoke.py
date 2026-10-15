"""Download two attributed NASA images for encoder mechanics, not M2 acceptance."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

from PIL import Image

SOURCES = [
    ("moon", "https://www.nasa.gov/wp-content/uploads/2023/03/as11-44-6552.jpeg",
     "5de2f18993a50c2a29c42c37a52f1b20c5e960911a863305007093789a99730a", "NASA/Apollo 11"),
    ("earth", "https://images-assets.nasa.gov/image/PIA18033/PIA18033~small.jpg",
     "5b47eb38676ad790c2e6245ed1025c88ee31ad23f55e516066865f2f601046ec",
     "NASA/NOAA/GSFC/Suomi NPP/VIIRS/Norman Kuring"),
]
POLICY = "https://www.nasa.gov/nasa-brand-center/images-and-media/"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, default=Path("run-data/images/m2-smoke"))
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/m2-image-smoke-inputs.json"))
    args = parser.parse_args()
    args.image_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, url, expected, credit in SOURCES:
        path = args.image_dir / f"{name}.jpg"
        if not path.exists():
            with urlopen(url, timeout=60) as response:
                data = response.read(20_000_001)
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError("Upstream image bytes changed; review before accepting")
            path.write_bytes(data)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Local smoke image checksum differs")
        with Image.open(path) as image:
            image.verify()
        rows.append({"id": name, "image": str(path), "scene_group": name,
                     "source": url, "license": f"{credit}; NASA imagery usage guidelines {POLICY}"})
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(rows, indent=2) + "\n")
    print("Prepared two NASA images locally; not training pairs or a grounding benchmark")


if __name__ == "__main__":
    main()
