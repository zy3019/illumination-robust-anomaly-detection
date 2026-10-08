"""Development-only experiment runner. Final-test execution is deliberately absent."""
import argparse
import csv
import hashlib
import json
import os
import platform
import re
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torchvision
from PIL import Image

from .data import (file_hash, load_image, load_mask, perturb, save_json,
                   select_training, training_views, validate_manifest)
from .metrics import auroc, classification_metrics
from .model import PatchFeatures, greedy_coreset, predict

PROJECT = Path(__file__).resolve().parents[1]


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def phase_start(device):
    sync(device)
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    return time.perf_counter()


def phase_end(start, device):
    sync(device)
    return {"seconds": time.perf_counter() - start,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if str(device).startswith("cuda") else None,
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if str(device).startswith("cuda") else None}


def process_peak_memory():
    if platform.system() != "Windows":
        return None
    import ctypes
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in ["PeakWorkingSetSize", "WorkingSetSize",
            "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
            "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage"]]
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    value = Counters()
    value.cb = ctypes.sizeof(value)
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(value), value.cb):
        return None
    return value.PeakWorkingSetSize


def visual(image, anomaly_map, threshold, mask, target):
    rgb = np.uint8(np.clip(image.permute(1, 2, 0).numpy(), 0, 1) * 255)
    # Fixed scale per model, shared across clean and perturbed images.
    heat = np.clip(anomaly_map / max(threshold * 1.5, 1e-8), 0, 1)
    colors = np.stack([heat, 0.3 * (1 - heat), 1 - heat], axis=-1)
    overlay = np.uint8(np.clip(0.55 * rgb + 0.45 * colors * 255, 0, 255))
    mask_rgb = np.repeat((mask * 255)[..., None], 3, axis=-1)
    Image.fromarray(np.concatenate([rgb, overlay, mask_rgb], axis=1)).save(target)


def validate_config(config):
    if config["evaluation_split"] != "dev":
        raise ValueError("This runner only permits development evaluation; final-test protocol is not locked")
    if config["method"] not in {"A", "B", "C"}:
        raise ValueError("Method must be A, B or C")
    if config["image_size"] < 32 or config["memory_budget"] < 1 or config["projection_dim"] < 1:
        raise ValueError("Invalid dimensions or memory budget")
    if not 0 < config["threshold_quantile"] < 1:
        raise ValueError("Invalid threshold quantile")
    names = set()
    for condition in config["conditions"]:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", condition["name"]) or condition["name"] in names:
            raise ValueError("Condition names must be unique, safe filenames")
        names.add(condition["name"])
        if condition["factor"] <= 0 or not np.isfinite(condition["factor"]):
            raise ValueError("Invalid perturbation factor")
        perturb(torch.zeros(3, 2, 2), condition["operation"], condition["factor"])
    if "clean" not in names or not any(c["name"] == "clean" and c["operation"] == "clean" for c in config["conditions"]):
        raise ValueError("A clean reference condition is required")


def run(config, output):
    run_started = time.perf_counter()
    validate_config(config)
    root = PROJECT / config["data_root"]
    manifest_path = PROJECT / config["manifest"]
    weights_path = PROJECT / config["weights"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    train = select_training(manifest["rows"], config["category"], config["seed"], config["normal_images"])
    calibration = [r for r in manifest["rows"] if r["category"] == config["category"] and r["split"] == "calibration"]
    evaluation = [r for r in manifest["rows"] if r["category"] == config["category"] and r["split"] == "dev"]
    if not calibration or {r["label"] for r in evaluation} != {0, 1}:
        raise ValueError("Need clean calibration images and both normal/anomalous dev images")
    device = config["device"]
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Configured CUDA device unavailable; no silent CPU fallback")
    if not file_hash(weights_path).startswith("f37072fd"):
        raise ValueError("Expected official ResNet18 IMAGENET1K_V1 weights")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    save_json(output / "config.json", config)
    save_json(output / "sample_lists.json", {"train": train, "calibration": calibration, "dev": evaluation})
    source_hashes = {p.relative_to(PROJECT).as_posix(): file_hash(p)
                     for directory in ["anomaly", "scripts"] for p in sorted((PROJECT / directory).glob("*.py"))}
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT, capture_output=True, text=True)
    provenance = {
        "started_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "status": "development_pilot", "git_commit": git.stdout.strip() if git.returncode == 0 else None,
        "source_sha256": source_hashes, "manifest_sha256": file_hash(manifest_path),
        "weights_sha256": file_hash(weights_path), "python": platform.python_version(),
        "torch": torch.__version__, "torchvision": torchvision.__version__, "numpy": np.__version__,
        "device": str(device), "gpu": torch.cuda.get_device_name(device) if str(device).startswith("cuda") else None,
        "preprocessing": "PIL RGB, direct square resize BILINEAR, perturb [0,1], ImageNet normalization",
        "mask_resize": "NEAREST", "image_scoring": "maximum raw 1-NN patch distance; no reweighting",
        "map": "bilinear interpolation, align_corners=False, no Gaussian smoothing",
        "pixel_auroc": "pooled resized pixels across all dev images, including normal backgrounds",
        "timing": "batch=1 float32; 3 warmups; transfer+features+NN+map+CPU output; excludes disk I/O and PIL resize",
    }
    save_json(output / "provenance.json", provenance)
    extractor = PatchFeatures(weights_path, device)
    views = {"A": 0, "B": 3, "C": 1}[config["method"]]
    augmentations, candidates = [], []
    start = phase_start(device)
    for row in train:
        image = load_image(root, row, config["image_size"])
        for view, metadata in training_views(image, row, config["seed"], views):
            features, _ = extractor(view.unsqueeze(0).to(device))
            candidates.append(features[0].cpu())
            augmentations.append(metadata)
    candidates = torch.cat(candidates)
    resources = {"feature_extraction": phase_end(start, device)}
    resources["feature_extraction"]["process_peak_working_set_bytes_so_far"] = process_peak_memory()
    save_json(output / "training_views.json", augmentations)
    print(f"{len(train)} originals; {len(augmentations)} views; {len(candidates)} candidates", flush=True)
    start = phase_start(device)
    bank, indices = greedy_coreset(candidates, config["memory_budget"], config["seed"], device, config["projection_dim"])
    resources["coreset"] = phase_end(start, device)
    resources["coreset"]["process_peak_working_set_bytes_so_far"] = process_peak_memory()
    candidate_count = len(candidates)
    candidate_bytes = candidates.numel() * candidates.element_size()
    del candidates
    bank = bank.to(device)
    start = phase_start(device)
    calibration_scores = []
    for row in calibration:
        score, _ = predict(extractor, load_image(root, row, config["image_size"]).to(device), bank)
        calibration_scores.append({"path": row["path"], "score": score})
    scores = [r["score"] for r in calibration_scores]
    threshold = float(np.quantile(scores, config["threshold_quantile"], method="linear"))
    resources["calibration"] = phase_end(start, device)
    write_csv(output / "calibration_scores.csv", calibration_scores)
    threshold_record = {"threshold": threshold, "quantile": config["threshold_quantile"], "method": "linear",
                        "rule": "score > threshold", "calibration_images": len(scores),
                        "calibration_fpr": float(np.mean(np.asarray(scores) > threshold))}
    save_json(output / "threshold.json", threshold_record)
    torch.save({"memory_bank": bank.cpu(), "selected_indices": indices, "config": config,
                "threshold": threshold, "manifest_sha256": provenance["manifest_sha256"],
                "weights_sha256": provenance["weights_sha256"]}, output / "model.pt")
    # Repeated warmups are intentionally excluded from recorded per-image timings.
    warmup = load_image(root, evaluation[0], config["image_size"]).to(device)
    for _ in range(3):
        predict(extractor, warmup, bank)
    sync(device)
    # Stable microbenchmark: three fixed dev images, ten repeats each; excludes
    # disk, resize and reporting. Scores below are still computed only once per condition.
    benchmark_rows = []
    for row in evaluation[:3]:
        image = load_image(root, row, config["image_size"])
        durations = []
        for _ in range(10):
            sync(device)
            tick = time.perf_counter()
            predict(extractor, image.to(device), bank)
            sync(device)
            durations.append(time.perf_counter() - tick)
        benchmark_rows.append({"path": row["path"], "repeats": 10,
                               "seconds": durations, "median_seconds": float(np.median(durations))})
    save_json(output / "inference_benchmark.json", {
        "warmups": 3, "batch_size": 1, "precision": "float32", "images": benchmark_rows,
        "overall_median_seconds": float(np.median([v for r in benchmark_rows for v in r["seconds"]])),
        "scope": "CPU tensor -> GPU -> features -> 1NN -> map -> CPU; excludes file IO and resize",
    })
    metrics, all_scores = [], []
    start = phase_start(device)
    for condition in config["conditions"]:
        folder = output / "maps" / condition["name"]
        folder.mkdir(parents=True)
        labels, scores, pixel_labels, pixel_scores, timings = [], [], [], [], []
        shown_defects = set()
        for row in evaluation:
            image = perturb(load_image(root, row, config["image_size"]), condition["operation"], condition["factor"])
            mask = load_mask(root, row, config["image_size"])
            sync(device)
            tick = time.perf_counter()
            score, anomaly_map = predict(extractor, image.to(device), bank)
            sync(device)
            elapsed = time.perf_counter() - tick
            identifier = f'{row["defect"]}_{Path(row["path"]).stem}'
            np.save(folder / f"{identifier}.npy", anomaly_map)
            if row["defect"] not in shown_defects:
                visual(image, anomaly_map, threshold, mask, folder / f"{identifier}.png")
                shown_defects.add(row["defect"])
            labels.append(row["label"])
            scores.append(score)
            pixel_labels.append(mask.ravel())
            pixel_scores.append(anomaly_map.ravel())
            timings.append(elapsed)
            all_scores.append({"condition": condition["name"], "path": row["path"], "label": row["label"],
                               "defect": row["defect"], "score": score, "threshold": threshold,
                               "predicted_anomaly": int(score > threshold), "seconds": elapsed})
        result = {"condition": condition["name"], **classification_metrics(labels, scores, threshold),
                  "pixel_auroc": auroc(np.concatenate(pixel_labels), np.concatenate(pixel_scores)),
                  "inference_seconds_median": float(np.median(timings)), "timed_images": len(timings)}
        metrics.append(result)
        print(json.dumps(result), flush=True)
    resources["evaluation"] = phase_end(start, device)
    resources["evaluation"]["process_peak_working_set_bytes_so_far"] = process_peak_memory()
    clean = next(r for r in metrics if r["condition"] == "clean")
    for result in metrics:
        result["image_auroc_drop"] = clean["image_auroc"] - result["image_auroc"]
        result["fpr_increase"] = result["fpr"] - clean["fpr"]
    write_csv(output / "scores.csv", all_scores)
    write_csv(output / "metrics.csv", metrics)
    summary = {
        "status": "development_pilot_only", "category": config["category"], "method": config["method"],
        "seed": config["seed"], "training_originals": len(train), "augmented_copies": len(train) * views,
        "calibration_images": len(calibration), "dev_images": len(evaluation), "final_test_evaluated": False,
        "candidate_vectors": candidate_count, "candidate_bytes": candidate_bytes,
        "memory_vectors": len(bank), "memory_bytes": bank.numel() * bank.element_size(),
        "feature_dimension": bank.shape[1], "threshold": threshold_record, "resources": resources, "metrics": metrics,
        "run_wall_seconds": time.perf_counter() - run_started,
        "repeated_inference_median_seconds": float(np.median([v for r in benchmark_rows for v in r["seconds"]])),
    }
    save_json(output / "summary.json", summary)
    print(f"Completed: {output.resolve()}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/bottle_pilot.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--method", choices=["A", "B", "C"])
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.method:
        config["method"] = args.method
    run(config, args.output)


if __name__ == "__main__":
    main()
