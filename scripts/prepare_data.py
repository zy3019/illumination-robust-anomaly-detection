"""Create one immutable, hashed manifest shared by all model configurations."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anomaly.data import make_manifest, save_json, validate_manifest, file_hash, contained


def verify_existing(root, target, categories, seed):
    manifest = json.loads(Path(target).read_text(encoding="utf-8-sig"))
    validate_manifest(manifest)
    if manifest['split_seed'] != seed or {r['category'] for r in manifest['rows']} != set(categories):
        raise ValueError("Existing manifest categories/seed differ; do not overwrite the frozen manifest")
    for row in manifest['rows']:
        if file_hash(contained(root, row['path'])) != row['sha256']:
            raise ValueError("Changed image: " + row['path'])
        if row['mask'] and file_hash(contained(root, row['mask'])) != row['mask_sha256']:
            raise ValueError("Changed mask: " + row['mask'])
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/mvtec_ad")
    parser.add_argument("--categories", nargs="+", default=["bottle", "metal_nut", "tile"])
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--output", default="records/manifests/mvtec_v1.json")
    parser.add_argument("--verify-existing", action="store_true",
                        help="Verify saved split and all image/mask hashes without rewriting it")
    args = parser.parse_args()
    target = Path(args.output)
    if args.verify_existing:
        manifest = verify_existing(args.root, target, args.categories, args.seed)
        print(f"Verified {len(manifest['rows'])} images and their masks; manifest unchanged")
        return
    if target.exists():
        raise FileExistsError(f"Manifest already exists; use a new filename for a deliberate new protocol: {target}")
    manifest = make_manifest(args.root, args.categories, args.seed)
    validate_manifest(manifest)
    save_json(target, manifest)
    print(json.dumps(manifest["counts"], indent=2))
    print(f"Saved {target}")


if __name__ == "__main__":
    main()
