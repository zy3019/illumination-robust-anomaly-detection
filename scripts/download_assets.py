"""Download public assets into the project; validate the dataset before extraction."""
import argparse
import hashlib
import json
import tarfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_URL = ("https://www.mydrive.ch/shares/150996/b52ecdcbf521176e9db9c731f2304b27/"
            "download/420938113-1629960298/mvtec_anomaly_detection.tar.xz")
DATA_SHA256 = "cf4313b13603bec67abb49ca959488f7eedce2a9f7795ec54446c649ac98cd3d"
WEIGHTS_URL = "https://download.pytorch.org/models/resnet18-f37072fd.pth"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(url, path, expected):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not digest(path).startswith(expected):
            raise RuntimeError(f"Hash mismatch: {path}")
        print(f"Verified existing {path.name}", flush=True)
        return
    partial = path.with_suffix(path.suffix + ".part")
    total = 0
    last = time.monotonic()
    req = urllib.request.Request(url, headers={"User-Agent": "Python research project"})
    with urllib.request.urlopen(req, timeout=90) as response, partial.open("wb") as out:
        length = response.headers.get("Content-Length", "unknown")
        print(f"Downloading {path.name}; bytes={length}", flush=True)
        while chunk := response.read(1024 * 1024):
            out.write(chunk)
            total += len(chunk)
            if time.monotonic() - last > 20:
                print(f"Downloaded {total / 1024**2:.0f} MiB", flush=True)
                last = time.monotonic()
    checksum = digest(partial)
    if not checksum.startswith(expected):
        raise RuntimeError(f"Hash mismatch; retained {partial}; actual={checksum}")
    partial.replace(path)
    print(f"Verified {path.name}: {checksum}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-only", action="store_true")
    parser.add_argument("--categories", nargs="+", default=["bottle", "metal_nut", "tile"])
    args = parser.parse_args()
    download(WEIGHTS_URL, ROOT / "data/weights/resnet18-f37072fd.pth", "f37072fd")
    if args.weights_only:
        return
    archive = ROOT / "data/downloads/mvtec_anomaly_detection.tar.xz"
    download(DATA_URL, archive, DATA_SHA256)
    target = ROOT / "data/mvtec_ad"
    target.mkdir(parents=True, exist_ok=True)
    marker = target / "download_provenance.json"
    if marker.exists():
        existing = json.loads(marker.read_text(encoding="utf-8"))
        if existing["categories"] == args.categories:
            print("Dataset already extracted", flush=True)
            return
    # Streaming decompression; tarfile's data filter rejects traversal and unsafe links.
    count = 0
    with tarfile.open(archive, "r|xz") as tar:
        for member in tar:
            parts = Path(member.name).parts
            if parts and parts[0] in args.categories and member.isfile():
                tar.extract(member, target, filter="data")
                count += 1
                if count % 500 == 0:
                    print(f"Extracted {count} files", flush=True)
    for category in args.categories:
        if not (target / category / "train/good").is_dir():
            raise RuntimeError(f"Missing extracted category: {category}")
    marker.write_text(json.dumps({
        "url": DATA_URL, "sha256": DATA_SHA256,
        "source": "https://github.com/open-edge-platform/anomalib/blob/main/src/anomalib/data/datamodules/image/mvtecad.py",
        "license": "CC BY-NC-SA 4.0", "categories": args.categories,
        "extracted_files": count,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Dataset ready: {target}", flush=True)


if __name__ == "__main__":
    main()
