"""Fetch the immutable Anomalib source used by the numerical audit."""
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "592aafc373091dda8206510a0853eb11c8e4099d"
PATHS = [
    "pyproject.toml", "LICENSE",
    "src/anomalib/models/image/patchcore/torch_model.py",
    "src/anomalib/models/image/patchcore/anomaly_map.py",
    "src/anomalib/models/image/patchcore/lightning_model.py",
    "src/anomalib/models/components/sampling/k_center_greedy.py",
    "src/anomalib/models/components/dimensionality_reduction/random_projection.py",
    "src/anomalib/models/components/feature_extractors/timm.py",
    "src/anomalib/models/components/filters/blur.py",
]


def main():
    folder = ROOT / "data/reference/anomalib-v2.2.0"
    hashes = {}
    for relative in PATHS:
        url = f"https://raw.githubusercontent.com/open-edge-platform/anomalib/{COMMIT}/{relative}"
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
        path = folder / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        hashes[relative] = hashlib.sha256(data).hexdigest()
    (folder / "source.json").write_text(json.dumps({
        "tag": "v2.2.0", "commit": COMMIT, "paths": PATHS, "sha256": hashes,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Pinned reference saved: {folder}")


if __name__ == "__main__":
    main()
