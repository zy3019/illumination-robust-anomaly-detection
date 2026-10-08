"""Sequential, resumable development jobs with immutable plan and source snapshot."""
import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from anomaly.data import file_hash, save_json, select_training
from anomaly.run import validate_config


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, value):
    temporary = Path(path).with_suffix(".tmp")
    save_json(temporary, value)
    temporary.replace(path)


def core_hashes():
    return {p.relative_to(ROOT).as_posix(): file_hash(p) for p in sorted((ROOT / "anomaly").glob("*.py"))}


def build_jobs(matrix, base):
    jobs = []
    for category in matrix["categories"]:
        for count in matrix["normal_images"]:
            for seed in matrix["seeds"]:
                for method in matrix["methods"]:
                    config = {**base, "category": category, "normal_images": count, "seed": seed,
                              "method": method, "conditions": matrix["conditions"], "memory_budget": matrix["memory_budget"]}
                    validate_config(config)
                    jobs.append({"id": f"{category}_n{count}_{method}_seed{seed}", "config": config})
    if len({job["id"] for job in jobs}) != len(jobs):
        raise ValueError("Duplicate jobs")
    return jobs


def verify_run(path, config, expected_core):
    path = Path(path)
    if read(path / "config.json") != config:
        raise ValueError(f"Config mismatch: {path}")
    summary = read(path / "summary.json")
    provenance = read(path / "provenance.json")
    if any(provenance["source_sha256"].get(k) != v for k, v in expected_core.items()):
        raise ValueError("Core implementation changed")
    if provenance["manifest_sha256"] != file_hash(ROOT / config["manifest"]):
        raise ValueError("Manifest changed")
    if summary["final_test_evaluated"] or summary["memory_vectors"] != config["memory_budget"]:
        raise ValueError("Invalid evaluation split or bank capacity")
    manifest = read(ROOT / config["manifest"])
    samples = read(path / "sample_lists.json")
    if samples["train"] != select_training(manifest["rows"], config["category"], config["seed"], config["normal_images"]):
        raise ValueError("Training sample mismatch")
    for key, split in [("calibration", "calibration"), ("dev", "dev")]:
        expected = [r for r in manifest["rows"] if r["category"] == config["category"] and r["split"] == split]
        if samples[key] != expected:
            raise ValueError(f"{key} samples mismatch")
    with (path / "scores.csv").open(encoding="utf-8") as stream:
        scores = list(csv.DictReader(stream))
    pairs = {(c["name"], r["path"]) for c in config["conditions"] for r in samples["dev"]}
    if len(scores) != len(pairs) or {(r["condition"], r["path"]) for r in scores} != pairs:
        raise ValueError("Missing or duplicate scores")
    if any(float(r["threshold"]) != summary["threshold"]["threshold"] for r in scores):
        raise ValueError("Threshold differs across conditions")
    if len(summary["metrics"]) != len(config["conditions"]) or {r["condition"] for r in summary["metrics"]} != {c["name"] for c in config["conditions"]}:
        raise ValueError("Incomplete metric conditions")
    for condition in config["conditions"]:
        for row in samples["dev"]:
            map_path = path / "maps" / condition["name"] / f'{row["defect"]}_{Path(row["path"]).stem}.npy'
            if not map_path.is_file():
                raise ValueError(f"Missing anomaly map: {map_path}")
    benchmark = read(path / "inference_benchmark.json")
    if len(benchmark["images"]) != 3 or any(len(r["seconds"]) != 10 for r in benchmark["images"]):
        raise ValueError("Incomplete inference benchmark")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", default="configs/dev_matrix_v1.json")
    args = parser.parse_args()
    matrix = read(ROOT / args.matrix)
    base = read(ROOT / matrix["base_config"])
    jobs = build_jobs(matrix, base)
    folder = ROOT / "outputs" / matrix["name"]
    record = ROOT / "records" / matrix["name"]
    folder.mkdir(parents=True, exist_ok=True)
    record.mkdir(parents=True, exist_ok=True)
    lock = folder / "RUNNING.lock"
    # Exclusive creation prevents competing schedulers. Remove a stale lock only
    # after verifying the previous scheduler is no longer alive.
    import os
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    try:
        plan = {"matrix": matrix, "jobs": jobs, "core_sha256": core_hashes(),
                "manifest_sha256": file_hash(ROOT / base["manifest"]),
                "weights_sha256": file_hash(ROOT / base["weights"])}
        plan_path = record / "plan.json"
        if plan_path.exists():
            if read(plan_path) != plan:
                raise ValueError("Saved plan differs; create a new matrix name for changed code/config")
        else:
            save_json(plan_path, plan)
            for relative in [*plan["core_sha256"], "scripts/run_dev_matrix.py", args.matrix]:
                target = record / "source" / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / relative, target)
        status_path = folder / "status.json"
        status = read(status_path) if status_path.exists() else {"started_at": datetime.now(timezone(timedelta(hours=8))).isoformat(), "jobs": {}}
        for position, job in enumerate(jobs, start=1):
            if core_hashes() != plan["core_sha256"]:
                raise ValueError("Core code modified during execution")
            previous = status["jobs"].get(job["id"], {})
            if previous.get("status") == "complete":
                verify_run(ROOT / previous["output"], job["config"], plan["core_sha256"])
                print(f'[{position}/{len(jobs)}] Verified completed {job["id"]}', flush=True)
                continue
            attempt = 1
            while (folder / f'{job["id"]}_attempt{attempt:02}').exists():
                attempt += 1
            target = folder / f'{job["id"]}_attempt{attempt:02}'
            config_path = record / "configs" / f'{job["id"]}.json'
            save_json(config_path, job["config"])
            log_path = folder / f'{job["id"]}_attempt{attempt:02}.log'
            entry = {"status": "running", "output": target.relative_to(ROOT).as_posix(), "attempt": attempt,
                     "log": log_path.relative_to(ROOT).as_posix()}
            status["jobs"][job["id"]] = entry
            atomic_json(status_path, status)
            print(f'[{position}/{len(jobs)}] Running {job["id"]}', flush=True)
            started = time.perf_counter()
            with log_path.open("w", encoding="utf-8") as log:
                result = subprocess.run([sys.executable, "-u", "-m", "anomaly.run", "--config", str(config_path),
                                         "--output", str(target)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            entry["process_wall_seconds"] = time.perf_counter() - started
            if result.returncode:
                entry["status"] = "failed"
                atomic_json(status_path, status)
                raise RuntimeError(f'Job failed; see {log_path}')
            summary = verify_run(target, job["config"], plan["core_sha256"])
            entry["status"] = "complete"
            entry["artifact_bytes"] = sum(p.stat().st_size for p in target.rglob("*") if p.is_file())
            entry["summary_sha256"] = file_hash(target / "summary.json")
            atomic_json(status_path, status)
            save_json(record / "summaries" / f'{job["id"]}.json', summary)
            print(f'[{position}/{len(jobs)}] Complete {job["id"]}: {entry["process_wall_seconds"]:.1f}s', flush=True)
        status["status"] = "complete"
        status["finished_at"] = datetime.now(timezone(timedelta(hours=8))).isoformat()
        atomic_json(status_path, status)
        save_json(record / "status.json", status)
        print(f"All {len(jobs)} development jobs complete", flush=True)
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
