"""Numerical audit against the installed, pinned Anomalib 2.2.0 model.

Run with .venv-audit, not the project runtime. This uses actual upstream classes,
with identical input tensors and transplanted, verified torchvision weights.
It does not run Lightning's Engine or its automatic post-processing.
"""
import argparse
import csv
import importlib.metadata
import inspect
import json
import math
import os
import platform
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "data/cache/matplotlib"))
sys.path.insert(0, str(ROOT))

import anomalib
import numpy as np
import torch
import torch.nn.functional as F
from anomalib.models.components.sampling.k_center_greedy import KCenterGreedy
from anomalib.models.image.patchcore.torch_model import PatchcoreModel
from sklearn.metrics import roc_auc_score
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models import resnet18

from anomaly.data import file_hash, load_image, load_mask, perturb, save_json, training_views
from anomaly.metrics import auroc, classification_metrics
from anomaly.model import PatchFeatures, greedy_coreset, nearest_distances, predict

REFERENCE_COMMIT = "592aafc373091dda8206510a0853eb11c8e4099d"
FEATURE_ATOL = 1e-6
SCORE_ATOL = 5e-4
SCORE_RTOL = 2e-5


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def maximum_error(x, y):
    return float((x - y).abs().max())


def check_source():
    if anomalib.__version__ != "2.2.0":
        raise RuntimeError("Audit requires anomalib==2.2.0")
    snapshot = ROOT / "data/reference/anomalib-v2.2.0"
    source = read_json(snapshot / "source.json")
    if source["commit"] != REFERENCE_COMMIT:
        raise RuntimeError("Unexpected upstream commit")
    package = Path(inspect.getfile(anomalib)).parent
    evidence = {}
    for relative in source["paths"]:
        if not relative.startswith("src/anomalib/"):
            continue
        installed = package / relative.removeprefix("src/anomalib/")
        saved = snapshot / relative
        if installed.read_text(encoding="utf-8") != saved.read_text(encoding="utf-8"):
            raise RuntimeError(f"Installed source differs from pinned source: {relative}")
        evidence[relative] = {"installed_sha256": file_hash(installed), "snapshot_sha256": file_hash(saved)}
    return evidence


def make_reference(weights_path, device):
    reference = PatchcoreModel(layers=["layer2", "layer3"], backbone="resnet18",
                              pre_trained=False, num_neighbors=1).eval().to(device)
    state = torch.load(weights_path, map_location=device, weights_only=True)
    target = reference.feature_extractor.feature_extractor
    transferred = {}
    for key, value in target.state_dict().items():
        if key in state:
            transferred[key] = state[key]
        elif key.endswith(".num_batches_tracked"):
            # Original torchvision checkpoint predates these counters; its native
            # loader inserts zero. They are inactive in eval mode.
            transferred[key] = torch.zeros_like(value)
        else:
            raise KeyError(f"Missing pretrained parameter: {key}")
    target.load_state_dict(transferred, strict=True)
    for key, value in target.state_dict().items():
        torch.testing.assert_close(value, transferred[key], atol=0, rtol=0)
    reference.requires_grad_(False)
    return reference


def embeddings(reference, normalized):
    raw = reference.feature_extractor(normalized)
    pooled = {key: reference.feature_pooler(value) for key, value in raw.items()}
    combined = reference.generate_embedding(pooled)
    return reference.reshape_embedding(combined), raw


@torch.inference_mode()
def audit_method(run_path, device, output):
    config = read_json(run_path / "config.json")
    if config["evaluation_split"] != "dev":
        raise ValueError("Audit is development-only")
    samples = read_json(run_path / "sample_lists.json")
    if any(r["split"] != "dev" for r in samples["dev"]):
        raise ValueError("Non-development evaluation rows")
    checkpoint = torch.load(run_path / "model.pt", map_location="cpu", weights_only=True)
    if checkpoint["config"] != config:
        raise ValueError("Checkpoint/config mismatch")
    weights = ROOT / config["weights"]
    if file_hash(weights) != checkpoint["weights_sha256"]:
        raise ValueError("Weights changed")
    if file_hash(ROOT / config["manifest"]) != checkpoint["manifest_sha256"]:
        raise ValueError("Manifest changed")
    local = PatchFeatures(weights, device)
    reference = make_reference(weights, device)
    # Independently trace torchvision for raw-layer verification.
    tv_model = resnet18(weights=None)
    tv_model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    traced = create_feature_extractor(tv_model.eval(), return_nodes={"layer2": "layer2", "layer3": "layer3"}).to(device).eval()
    root = ROOT / config["data_root"]
    mean, std = local.mean, local.std
    candidate_parts = []
    errors = {"raw_layer_max_abs": 0., "embedding_max_abs": 0., "nearest_max_abs": 0.,
              "max_image_score_abs": 0., "legacy_score_max_abs": 0., "raw_map_max_abs": 0.,
              "smoothed_map_max_abs_difference": 0., "scoring_9_max_abs_difference": 0.}
    feature_cases = 0

    def features(image):
        nonlocal feature_cases
        image = image.unsqueeze(0).to(device)
        normalized = (image - mean) / std
        own, shape = local(image)
        other, raw = embeddings(reference, normalized)
        independent = traced(normalized)
        for layer in ["layer2", "layer3"]:
            torch.testing.assert_close(raw[layer], independent[layer], atol=FEATURE_ATOL, rtol=1e-6)
            errors["raw_layer_max_abs"] = max(errors["raw_layer_max_abs"], maximum_error(raw[layer], independent[layer]))
        torch.testing.assert_close(own[0], other, atol=FEATURE_ATOL, rtol=1e-6)
        errors["embedding_max_abs"] = max(errors["embedding_max_abs"], maximum_error(own[0], other))
        feature_cases += 1
        return own[0], shape, normalized

    views = {"A": 0, "B": 3, "C": 1}[config["method"]]
    for row in samples["train"]:
        for image, _ in training_views(load_image(root, row, config["image_size"]), row, config["seed"], views):
            own, _, _ = features(image)
            candidate_parts.append(own.cpu())
    candidates = torch.cat(candidate_parts)
    bank = checkpoint["memory_bank"].to(device)
    rebuilt, indices = greedy_coreset(candidates, config["memory_budget"], config["seed"], device, config["projection_dim"])
    torch.testing.assert_close(rebuilt, bank.cpu(), atol=0, rtol=0)
    torch.testing.assert_close(indices, checkpoint["selected_indices"], atol=0, rtol=0)
    # The local coreset is intentionally different: quantify rather than force equality.
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    ratio = config["memory_budget"] / len(candidates)
    if int(len(candidates) * ratio) != config["memory_budget"]:
        ratio = math.nextafter(ratio, math.inf)
    sampler = KCenterGreedy(candidates.to(device), sampling_ratio=ratio)
    upstream_indices = sampler.select_coreset_idxs()
    if len(upstream_indices) != config["memory_budget"] or len(set(upstream_indices)) != len(upstream_indices):
        raise AssertionError("Upstream coreset capacity or uniqueness mismatch")
    upstream_bank = candidates[upstream_indices].to(device)
    coreset = {"candidate_vectors": len(candidates), "local_projection_dim": config["projection_dim"],
               "upstream_projection_dim": int(sampler.model.n_components), "upstream_ratio": ratio,
               "actual_upstream_capacity": len(upstream_indices), "local_capacity": len(bank),
               "overlapping_indices": len(set(indices.tolist()) & set(upstream_indices)),
               "local_rebuild_exact": True}
    del sampler, candidates, candidate_parts
    torch.save({"memory_bank": upstream_bank.cpu(), "indices": upstream_indices,
                "config": config, "reference_commit": REFERENCE_COMMIT}, output / f'upstream_bank_{config["method"]}.pt')
    with (run_path / "scores.csv").open(encoding="utf-8") as stream:
        legacy = {(r["condition"], r["path"]): float(r["score"]) for r in csv.DictReader(stream)}
    with (run_path / "calibration_scores.csv").open(encoding="utf-8") as stream:
        legacy_calibration = {r["path"]: float(r["score"]) for r in csv.DictReader(stream)}
    groups = {"local": [], "same_bank_k1": [], "same_bank_k9": [], "upstream_bank_k9": []}

    def score(image):
        own, (h, w), normalized = features(image)
        distances = nearest_distances(own, bank)
        reference.memory_bank = bank
        ref_distances, locations = reference.nearest_neighbors(own, 1)
        torch.testing.assert_close(distances, ref_distances, atol=SCORE_ATOL, rtol=SCORE_RTOL)
        errors["nearest_max_abs"] = max(errors["nearest_max_abs"], maximum_error(distances, ref_distances))
        local_score, local_map = predict(local, image.to(device), bank)
        reference.num_neighbors = 1
        ref_one = reference(normalized)
        torch.testing.assert_close(torch.tensor(local_score, device=device), ref_one.pred_score[0], atol=SCORE_ATOL, rtol=SCORE_RTOL)
        errors["max_image_score_abs"] = max(errors["max_image_score_abs"], abs(local_score - float(ref_one.pred_score[0])))
        raw_map = F.interpolate(ref_distances.reshape(1, 1, h, w), size=image.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        torch.testing.assert_close(torch.from_numpy(local_map).to(device), raw_map, atol=SCORE_ATOL, rtol=SCORE_RTOL)
        errors["raw_map_max_abs"] = max(errors["raw_map_max_abs"], maximum_error(torch.from_numpy(local_map).to(device), raw_map))
        smoothed_map = ref_one.anomaly_map[0, 0].cpu().numpy()
        errors["smoothed_map_max_abs_difference"] = max(errors["smoothed_map_max_abs_difference"], float(np.abs(local_map - smoothed_map).max()))
        reference.num_neighbors = 9
        weighted = float(reference.compute_anomaly_score(ref_distances.unsqueeze(0), locations.unsqueeze(0), own)[0])
        # Validate the actual full reference forward with weighting enabled.
        ref_nine = reference(normalized)
        torch.testing.assert_close(torch.tensor(weighted, device=device), ref_nine.pred_score[0], atol=SCORE_ATOL, rtol=SCORE_RTOL)
        errors["scoring_9_max_abs_difference"] = max(errors["scoring_9_max_abs_difference"], abs(local_score - weighted))
        reference.memory_bank = upstream_bank
        upstream = reference(normalized)
        values = {"local": local_score, "same_bank_k1": float(ref_one.pred_score[0]),
                  "same_bank_k9": weighted, "upstream_bank_k9": float(upstream.pred_score[0])}
        maps = {"local": local_map, "same_bank_k1": smoothed_map, "same_bank_k9": smoothed_map,
                "upstream_bank_k9": upstream.anomaly_map[0, 0].cpu().numpy()}
        return values, maps

    calibration_rows = []
    for row in samples["calibration"]:
        values, _ = score(load_image(root, row, config["image_size"]))
        error = abs(values["local"] - legacy_calibration[row["path"]])
        if error > SCORE_ATOL + SCORE_RTOL * abs(legacy_calibration[row["path"]]):
            raise AssertionError("Local calibration no longer reproduces stored run")
        errors["legacy_score_max_abs"] = max(errors["legacy_score_max_abs"], error)
        calibration_rows.append({"path": row["path"], **values})
        for key in groups:
            groups[key].append(values[key])
    thresholds = {key: float(np.quantile(values, config["threshold_quantile"], method="linear")) for key, values in groups.items()}
    if abs(thresholds["local"] - checkpoint["threshold"]) > SCORE_ATOL:
        raise AssertionError("Local threshold changed")
    summaries, score_rows = [], []
    for condition in config["conditions"]:
        scores = {key: [] for key in groups}
        maps = {key: [] for key in groups}
        labels, masks = [], []
        for row in samples["dev"]:
            image = perturb(load_image(root, row, config["image_size"]), condition["operation"], condition["factor"])
            values, anomaly_maps = score(image)
            expected = legacy[(condition["name"], row["path"])]
            error = abs(values["local"] - expected)
            if error > SCORE_ATOL + SCORE_RTOL * abs(expected):
                raise AssertionError("Local dev score no longer reproduces stored run")
            errors["legacy_score_max_abs"] = max(errors["legacy_score_max_abs"], error)
            labels.append(row["label"])
            masks.append(load_mask(root, row, config["image_size"]).ravel())
            for key in groups:
                scores[key].append(values[key])
                maps[key].append(anomaly_maps[key].ravel())
            score_rows.append({"condition": condition["name"], "path": row["path"], "label": row["label"], **values})
        all_masks = np.concatenate(masks)
        for key in groups:
            pixel_scores = np.concatenate(maps[key])
            result = classification_metrics(labels, scores[key], thresholds[key])
            pixel_auc = auroc(all_masks, pixel_scores)
            if abs(pixel_auc - roc_auc_score(all_masks, pixel_scores)) > 1e-12:
                raise AssertionError("Pixel AUROC differs from sklearn")
            if abs(result["image_auroc"] - roc_auc_score(labels, scores[key])) > 1e-12:
                raise AssertionError("Image AUROC differs from sklearn")
            summaries.append({"condition": condition["name"], "profile": key, **result, "pixel_auroc": pixel_auc})
    for name, rows in [("scores", score_rows), ("calibration", calibration_rows), ("metrics", summaries)]:
        with (output / f'{config["method"]}_{name}.csv').open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return {"method": config["method"], "device": device, "feature_cases": feature_cases,
            "scored_cases": len(calibration_rows) + len(score_rows), "errors": errors,
            "coreset": coreset, "thresholds": thresholds, "metrics": summaries,
            "inputs": {"config_sha256": file_hash(run_path / "config.json"),
                       "model_sha256": file_hash(run_path / "model.pt"),
                       "manifest_sha256": checkpoint["manifest_sha256"]}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="outputs/baseline_audit_v1")
    args = parser.parse_args()
    source = check_source()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=False)
    results = []
    for method in ["A", "B"]:
        result = audit_method(ROOT / f"outputs/bottle_{method}_seed42_n5", args.device, output)
        results.append(result)
        print(json.dumps({key: result[key] for key in ["method", "feature_cases", "scored_cases", "errors", "coreset"]}), flush=True)
    report = {"status": "passed", "timestamp": datetime.now(timezone(timedelta(hours=8))).isoformat(),
              "reference": {"version": "2.2.0", "commit": REFERENCE_COMMIT, "source": source},
              "tolerances": {"features_atol": FEATURE_ATOL, "features_rtol": 1e-6, "scores_atol": SCORE_ATOL, "scores_rtol": SCORE_RTOL},
              "scope": "Actual Anomalib PyTorch model; shared RGB resize/input/weights; local and upstream banks; no Lightning Engine or automatic postprocessor",
              "python": platform.python_version(), "dependencies": {name: importlib.metadata.version(name) for name in
                  ["anomalib", "torch", "torchvision", "timm", "kornia", "scikit-learn", "numpy"]},
              "audit_script_sha256": file_hash(Path(__file__)),
              "local_source_sha256": {p.relative_to(ROOT).as_posix(): file_hash(p) for p in (ROOT / "anomaly").glob("*.py")},
              "final_test_evaluated": False, "results": results}
    save_json(output / "audit.json", report)
    print(f"Audit passed: {output / 'audit.json'}", flush=True)


if __name__ == "__main__":
    main()
