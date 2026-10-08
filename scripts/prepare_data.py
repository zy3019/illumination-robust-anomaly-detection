"""Create one immutable, hashed manifest shared by all model configurations."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anomaly.data import make_manifest, save_json, validate_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/mvtec_ad")
    parser.add_argument("--categories", nargs="+", default=["bottle", "metal_nut", "tile"])
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--output", default="records/manifests/mvtec_v1.json")
    args = parser.parse_args()
    target = Path(args.output)
    if target.exists():
        raise FileExistsError(f"Manifest already exists; use a new filename for a deliberate new protocol: {target}")
    manifest = make_manifest(args.root, args.categories, args.seed)
    validate_manifest(manifest)
    save_json(target, manifest)
    print(json.dumps(manifest["counts"], indent=2))
    print(f"Saved {target}")


if __name__ == "__main__":
    main()
