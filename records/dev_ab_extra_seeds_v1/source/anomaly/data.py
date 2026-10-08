"""File-level MVTec splits and deterministic nested normal-image budgets."""
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def seeded_rng(seed, namespace):
    value = hashlib.sha256(f"{seed}:{namespace}".encode()).digest()
    return random.Random(int.from_bytes(value[:8], "big"))


def contained(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Path escapes dataset root: {relative}")
    return path


def make_manifest(root, categories, seed=20261005):
    root = Path(root).resolve()
    rows = []
    for category in categories:
        category_root = contained(root, category)
        normal = sorted((category_root / "train/good").glob("*.png"))
        if len(normal) < 32:
            raise ValueError(f"{category}: need >=32 normal train images for 25-image budget plus calibration")
        seeded_rng(seed, f"{category}/train").shuffle(normal)
        calibration_count = max(1, round(len(normal) * 0.2))
        groups = [(p, "calibration" if i < calibration_count else "train_pool", "good")
                  for i, p in enumerate(normal)]
        defect_dirs = sorted((category_root / "test").glob("*"))
        if not any(d.name == "good" for d in defect_dirs):
            raise ValueError(f"Missing normal test images for {category}")
        if not any(d.is_dir() and d.name != "good" for d in defect_dirs):
            raise ValueError(f"Missing anomalous test images for {category}")
        for directory in defect_dirs:
            if not directory.is_dir():
                continue
            paths = sorted(directory.glob("*.png"))
            if len(paths) < 2:
                raise ValueError(f"Too few images to split defect stratum: {directory}")
            seeded_rng(seed, f"{category}/test/{directory.name}").shuffle(paths)
            n_dev = min(len(paths) - 1, max(1, round(len(paths) * 0.2)))
            groups.extend((p, "dev" if i < n_dev else "final_test", directory.name)
                          for i, p in enumerate(paths))
        for path, split, defect in groups:
            mask = category_root / "ground_truth" / defect / f"{path.stem}_mask.png"
            if defect != "good" and not mask.is_file():
                raise FileNotFoundError(mask)
            rows.append({
                "category": category, "path": path.relative_to(root).as_posix(),
                "split": split, "defect": defect, "label": int(defect != "good"),
                "mask": mask.relative_to(root).as_posix() if defect != "good" else None,
                "sha256": file_hash(path),
                "mask_sha256": file_hash(mask) if defect != "good" else None,
            })
    rows.sort(key=lambda row: row["path"])
    counts = Counter(f'{r["category"]}/{r["split"]}/{r["defect"]}' for r in rows)
    return {"schema_version": 1, "split_seed": seed,
            "protocol": "custom dev/final split; 20% calibration; 20% dev per test defect type",
            "counts": dict(sorted(counts.items())), "rows": rows}


def validate_manifest(manifest):
    seen = set()
    hashes = {}
    for row in manifest["rows"]:
        if row["path"] in seen:
            raise ValueError(f'Duplicate manifest path: {row["path"]}')
        seen.add(row["path"])
        if row["split"] not in {"train_pool", "calibration", "dev", "final_test"}:
            raise ValueError("Invalid split")
        if row["label"] != int(row["defect"] != "good"):
            raise ValueError("Inconsistent anomaly label")
        if row["split"] in {"train_pool", "calibration"} and row["label"] != 0:
            raise ValueError("Training and calibration must be normal-only")
        previous = hashes.get((row["category"], row["sha256"]))
        if previous is not None and previous != row["split"]:
            raise ValueError(f'Identical image content crosses splits: {row["path"]}')
        hashes[(row["category"], row["sha256"])] = row["split"]


def select_training(rows, category, seed, count):
    pool = sorted((r for r in rows if r["category"] == category and r["split"] == "train_pool"),
                  key=lambda row: row["path"])
    seeded_rng(seed, f"{category}/training_order").shuffle(pool)
    n = len(pool) if count == "all" else int(count)
    if n < 1 or n > len(pool):
        raise ValueError(f"Requested {n} images from training pool of {len(pool)}")
    return pool[:n]


def load_image(root, row, size):
    path = contained(root, row["path"])
    if file_hash(path) != row["sha256"]:
        raise ValueError(f"Image changed since manifest creation: {path}")
    with Image.open(path) as im:
        array = np.array(im.convert("RGB").resize((size, size), Image.Resampling.BILINEAR), copy=True)
    return torch.from_numpy(array).permute(2, 0, 1).float().div(255)


def load_mask(root, row, size):
    if row["mask"] is None:
        return np.zeros((size, size), dtype=np.uint8)
    path = contained(root, row["mask"])
    if file_hash(path) != row["mask_sha256"]:
        raise ValueError(f"Mask changed since manifest creation: {path}")
    with Image.open(path) as im:
        return (np.asarray(im.convert("L").resize((size, size), Image.Resampling.NEAREST)) > 0).astype(np.uint8)


def perturb(image, operation="clean", factor=1.0):
    if operation == "clean":
        if factor != 1:
            raise ValueError("Clean condition requires factor=1")
        return image
    if operation == "brightness":
        return (image * factor).clamp(0, 1)
    if operation == "contrast":
        mean = image.mean(dim=(-2, -1), keepdim=True)
        return ((image - mean) * factor + mean).clamp(0, 1)
    raise ValueError(f"Unknown operation: {operation}")


def training_views(image, row, seed, views):
    yield image, {"path": row["path"], "view": 0, "operation": "clean", "factor": 1.0}
    rng = seeded_rng(seed, f'augmentation/{row["path"]}')
    for view in range(1, views + 1):
        operation = rng.choice(["brightness", "contrast"])
        factor = rng.uniform(0.8, 1.2)
        yield perturb(image, operation, factor), {
            "path": row["path"], "view": view, "operation": operation, "factor": factor}


def save_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
