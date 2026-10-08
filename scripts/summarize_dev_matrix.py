"""Verify all development jobs and build tables, figures and cost projections."""
import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "data/cache/matplotlib"))
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from anomaly.data import file_hash, save_json
from scripts.run_dev_matrix import verify_run


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(matrix_name):
    record = ROOT / "records" / matrix_name
    plan = read(record / "plan.json")
    status = read(ROOT / "outputs" / matrix_name / "status.json")
    if status.get("status") != "complete":
        raise ValueError("Cannot summarize an incomplete matrix")
    manifest = read(ROOT / plan["jobs"][0]["config"]["manifest"])
    counts = defaultdict(lambda: defaultdict(int))
    for row in manifest["rows"]:
        counts[row["category"]][row["split"]] += 1
        if row["split"] == "dev":
            counts[row["category"]]["dev_anomalous" if row["label"] else "dev_normal"] += 1
    resources, all_metrics, units = [], [], {}
    originals = {}
    for job in plan["jobs"]:
        entry = status["jobs"][job["id"]]
        if entry["status"] != "complete":
            raise ValueError(f'Incomplete job: {job["id"]}')
        folder = ROOT / entry["output"]
        if file_hash(folder / "summary.json") != entry["summary_sha256"]:
            raise ValueError("Summary changed after execution")
        summary = verify_run(folder, job["config"], plan["core_sha256"])
        cfg = job["config"]
        key = (cfg["category"], str(cfg["normal_images"]), cfg["method"])
        if key in units:
            raise ValueError("This single-seed summary requires one run per category/budget/method")
        units[key] = (summary, cfg, entry)
        samples = read(folder / "sample_lists.json")
        sample_key = (cfg["category"], str(cfg["normal_images"]))
        if sample_key in originals and samples != originals[sample_key]:
            raise ValueError("A/B sample lists differ")
        originals[sample_key] = samples
        meta = {"category": cfg["category"], "sample_budget": str(cfg["normal_images"]),
                "training_originals": summary["training_originals"], "method": cfg["method"], "seed": cfg["seed"]}
        phases = summary["resources"]
        resources.append({**meta, "candidate_vectors": summary["candidate_vectors"], "candidate_mib": summary["candidate_bytes"] / 2**20,
                          "bank_vectors": summary["memory_vectors"], "bank_mib": summary["memory_bytes"] / 2**20,
                          "feature_seconds": phases["feature_extraction"]["seconds"], "coreset_seconds": phases["coreset"]["seconds"],
                          "calibration_seconds": phases["calibration"]["seconds"], "evaluation_seconds": phases["evaluation"]["seconds"],
                          "process_wall_seconds": entry["process_wall_seconds"],
                          "peak_cpu_working_set_gib": max(p.get("process_peak_working_set_bytes_so_far") or 0 for p in phases.values()) / 2**30,
                          "peak_torch_allocated_mib": max(p["cuda_peak_allocated_bytes"] or 0 for p in phases.values()) / 2**20,
                          "peak_torch_reserved_mib": max(p["cuda_peak_reserved_bytes"] or 0 for p in phases.values()) / 2**20,
                          "inference_ms": summary["repeated_inference_median_seconds"] * 1000,
                          "artifact_mib": entry["artifact_bytes"] / 2**20})
        all_metrics.extend({**meta, **metric} for metric in summary["metrics"])
    for category in plan["matrix"]["categories"]:
        nested = [originals[(category, str(n))]["train"] for n in [5, 10, 25, "all"]]
        for small, large in zip(nested, nested[1:]):
            if small != large[:len(small)]:
                raise ValueError("Training sets are not nested")
    pairs = []
    for category in plan["matrix"]["categories"]:
        for count in plan["matrix"]["normal_images"]:
            a = units[(category, str(count), "A")][0]
            b = units[(category, str(count), "B")][0]
            for am, bm in zip(a["metrics"], b["metrics"]):
                if am["condition"] != bm["condition"]:
                    raise ValueError("Condition order mismatch")
                pairs.append({"category": category, "sample_budget": str(count), "condition": am["condition"],
                              "B_minus_A_image_auroc": bm["image_auroc"] - am["image_auroc"],
                              "B_minus_A_pixel_auroc": bm["pixel_auroc"] - am["pixel_auroc"],
                              "B_minus_A_fpr": bm["fpr"] - am["fpr"], "B_minus_A_recall": bm["recall"] - am["recall"]})
    folder = ROOT / "reports" / matrix_name
    folder.mkdir(parents=True, exist_ok=True)
    write_csv(folder / "metrics.csv", all_metrics)
    write_csv(folder / "paired_differences.csv", pairs)
    write_csv(folder / "resources.csv", resources)
    # A/B 72 builds = 3 times this one-seed 24-run matrix. C is estimated from
    # its two views, interpolated between A=1 and B=4 views, only n=5,25.
    formal_seconds, formal_bytes = 0., 0.
    projection_rows = []
    for category in plan["matrix"]["categories"]:
        ratio = counts[category]["final_test"] / counts[category]["dev"]
        for count in plan["matrix"]["normal_images"]:
            selected = [r for r in resources if r["category"] == category and r["sample_budget"] == str(count)]
            for r in selected:
                # Includes startup, loading and calibration at their measured costs;
                # scales complete evaluation time (inference, IO, metric sorting).
                estimate = r["process_wall_seconds"] + r["evaluation_seconds"] * (ratio - 1)
                disk = r["artifact_mib"] * ratio  # conservative: scales even the fixed-size artifacts
                formal_seconds += 3 * estimate
                formal_bytes += 3 * disk * 2**20
                projection_rows.append({"category": category, "sample_budget": str(count), "method": r["method"],
                                        "runs": 3, "estimated_seconds_per_run": estimate, "eval_image_ratio": ratio})
            if count in [5, 25]:
                a = next(r for r in selected if r["method"] == "A")
                b = next(r for r in selected if r["method"] == "B")
                estimate_a = a["process_wall_seconds"] + a["evaluation_seconds"] * (ratio - 1)
                estimate_b = b["process_wall_seconds"] + b["evaluation_seconds"] * (ratio - 1)
                estimate = estimate_a + (estimate_b - estimate_a) / 3
                formal_seconds += 3 * estimate
                formal_bytes += 3 * (a["artifact_mib"] + (b["artifact_mib"] - a["artifact_mib"]) / 3) * ratio * 2**20
                projection_rows.append({"category": category, "sample_budget": str(count), "method": "C_estimated",
                                        "runs": 3, "estimated_seconds_per_run": estimate, "eval_image_ratio": ratio})
    write_csv(folder / "formal_cost_projection.csv", projection_rows)
    total_cases = sum(u[0]["dev_images"] * len(u[0]["metrics"]) for u in units.values())
    estimates = {"status": "development_only", "run_count": len(units), "seed_count": len(plan["matrix"]["seeds"]),
                 "total_scored_dev_views": total_cases, "total_independent_dev_images": sum(counts[c]["dev"] for c in plan["matrix"]["categories"]),
                 "measured_process_wall_seconds": sum(r["process_wall_seconds"] for r in resources),
                 "measured_artifact_bytes": sum(status["jobs"][j["id"]]["artifact_bytes"] for j in plan["jobs"]),
                 "max_cpu_working_set_gib": max(r["peak_cpu_working_set_gib"] for r in resources),
                 "max_torch_allocated_mib": max(r["peak_torch_allocated_mib"] for r in resources),
                 "max_torch_reserved_mib": max(r["peak_torch_reserved_mib"] for r in resources),
                 "projected_formal_builds": sum(r["runs"] for r in projection_rows),
                 "projected_formal_seconds_point": formal_seconds, "planning_seconds_with_50pct_buffer": formal_seconds * 1.5,
                 "projected_formal_artifact_bytes": formal_bytes,
                 "assumptions": ["same implementation, device, 224 resolution, capacity 1000, sequential execution",
                                 "A/B: 72 builds; C: 18 builds at n=5/25, cost interpolated at two training views",
                                 "evaluation time scaled by final/dev image counts, not a measured final evaluation",
                                 "metric sorting, cache, thermals and future implementation changes can alter costs",
                                 "50% planning buffer is a heuristic, not a confidence interval",
                                 "excludes data download, manual analysis, AU-PRO, new methods and report writing"],
                 "final_test_evaluated": False, "manifest_sha256": plan["manifest_sha256"]}
    save_json(folder / "resource_estimate.json", estimates)
    draw_figures(folder, units, resources, plan["matrix"]["categories"])
    render_report(folder, units, resources, pairs, counts, estimates, plan)
    print(json.dumps(estimates, indent=2))


def draw_figures(folder, units, resources, categories):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    colors = {"A": "#2563eb", "B": "#d97706"}
    budgets = ["5", "10", "25", "all"]
    fig, axes = plt.subplots(3, 3, figsize=(13, 10), layout="constrained")
    for column, category in enumerate(categories):
        for method in ["A", "B"]:
            summaries = [units[(category, n, method)][0] for n in budgets]
            clean = [next(r for r in s["metrics"] if r["condition"] == "clean") for s in summaries]
            series = [[r["image_auroc"] for r in clean],
                      [max(r["image_auroc_drop"] for r in s["metrics"] if r["condition"] != "clean") for s in summaries],
                      [max(r["fpr"] for r in s["metrics"] if r["condition"] != "clean") for s in summaries]]
            for row, values in enumerate(series):
                axes[row, column].plot(range(4), values, marker="o", color=colors[method], label=method)
                axes[row, column].set_xticks(range(4), ["5", "10", "25", f'all ({summaries[-1]["training_originals"]})'])
                axes[row, column].grid(alpha=0.2)
                axes[row, column].set_xlabel("Training originals (categorical spacing)")
        axes[0, column].set_title(category)
        axes[0, column].legend(title="Method")
        axes[0, column].set_ylim(0, 1.04)
        axes[1, column].axhline(0, color="gray", linewidth=0.8)
        axes[2, column].set_ylim(-0.02, 1.04)
    for row, label in enumerate(["Clean image AUROC", "Worst AUROC drop over 12 perturbations", "Worst FPR over 12 perturbations"]):
        axes[row, 0].set_ylabel(label)
    fig.suptitle("Development only | seed 42 | fixed bank 1,000 | no uncertainty estimate", fontsize=14)
    fig.savefig(folder / "development_results.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), layout="constrained")
    for column, category in enumerate(categories):
        for method in ["A", "B"]:
            values = [next(r for r in resources if r["category"] == category and r["sample_budget"] == n and r["method"] == method)
                      for n in budgets]
            axes[column].plot(range(4), [r["feature_seconds"] + r["coreset_seconds"] for r in values], marker="o", color=colors[method], label=method)
        axes[column].set_title(category)
        axes[column].set_xticks(range(4), budgets)
        axes[column].set_xlabel("Training originals (categorical spacing)")
        axes[column].grid(alpha=0.2)
        axes[column].legend()
    axes[0].set_ylabel("Feature extraction + coreset time (s)")
    fig.suptitle("One measured build per setting; runtime variation is not estimated")
    fig.savefig(folder / "build_cost.png", dpi=160)
    plt.close(fig)


def render_report(folder, units, resources, pairs, counts, estimates, plan):
    categories = plan["matrix"]["categories"]
    lines = ["# 三类开发实验与资源估算", "", "本轮为开发试验，seed=42，固定 1,000 个记忆库向量。未评估最终测试集。", "",
             "主要观察、强扰动样例复核和后续决策见 [结果解读](findings.md)。", "",
             f'完成 **{len(units)} 次建库、{len(units) * 13} 组条件指标、{estimates["total_scored_dev_views"]:,} 次开发图像评分**。'
             f'这些评分来自 {estimates["total_independent_dev_images"]} 张独立开发原图，同一图片的多个扰动不视为独立样本。', "",
             "## 数据与设置", "", "| 类别 | 可用正常训练池 | 正常校准 | 开发正常 / 异常 | 保留最终测试 |",
             "|---|---:|---:|---:|---:|"]
    for c in categories:
        lines.append(f'| {c} | {counts[c]["train_pool"]} | {counts[c]["calibration"]} | {counts[c]["dev_normal"]} / {counts[c]["dev_anomalous"]} | {counts[c]["final_test"]} |')
    lines += ["", "A 使用原图；B 每张原图另加 3 个训练光照视图。训练数为 5/10/25/全量，同一类别的 A/B 样本一致，四档集合已验证嵌套。",
              "测试条件为干净及亮度、对比度分别取 0.9/1.1/0.7/1.3/0.5/1.5，共 13 个条件。所有阈值只由各模型的干净正常校准分数确定。", "",
              "## 开发结果", "", "![Development results](development_results.png)", "",
              "下表的最差退化是在 12 个光照条件中取最大值，条件可能因模型而异；并非对所有光照的平均结论。", "",
              "| 类别 | 原图数 | 方法 | 干净图像 AUROC | 干净像素 AUROC | 干净误报 / 正常 | 干净检出 / 异常 | 最大图像 AUROC 下降 | 对应条件 |",
              "|---|---:|---|---:|---:|---:|---:|---:|---|"]
    for c in categories:
        for n in ["5", "10", "25", "all"]:
            for m in ["A", "B"]:
                s = units[(c, n, m)][0]
                clean = next(r for r in s["metrics"] if r["condition"] == "clean")
                worst = max((r for r in s["metrics"] if r["condition"] != "clean"), key=lambda r: r["image_auroc_drop"])
                lines.append(f'| {c} | {s["training_originals"]} | {m} | {clean["image_auroc"]:.4f} | {clean["pixel_auroc"]:.4f} | '
                             f'{clean["false_positives"]}/{clean["normal_images"]} | {clean["true_positives"]}/{clean["anomalous_images"]} | '
                             f'{worst["image_auroc_drop"]:.4f} | {worst["condition"]} |')
    lines += ["", "## A/B 配对汇总", "", "下表先在同一类别内对 12 个扰动条件等权平均，再对三个类别等权宏平均。它描述本轮所选条件，不代表真实光照发生频率。", "",
              "| 原图预算 | 干净 B−A 图像 AUROC（宏平均） | 扰动 B−A 图像 AUROC（宏平均） | 扰动 B−A FPR（宏平均） | 扰动 B−A recall（宏平均） |",
              "|---|---:|---:|---:|---:|"]
    for n in ["5", "10", "25", "all"]:
        clean = [r for r in pairs if r["sample_budget"] == n and r["condition"] == "clean"]
        disturbed = [r for r in pairs if r["sample_budget"] == n and r["condition"] != "clean"]
        values = [np.mean([r["B_minus_A_image_auroc"] for r in clean])]
        values += [np.mean([r[key] for r in disturbed]) for key in ["B_minus_A_image_auroc", "B_minus_A_fpr", "B_minus_A_recall"]]
        lines.append(f'| {n} | ' + " | ".join(f"{v:+.5f}" for v in values) + " |")
    lines += ["", "## 实测资源", "", "![Build cost](build_cost.png)", "",
              f'24 个独立子进程累计耗时 **{estimates["measured_process_wall_seconds"] / 60:.2f} 分钟**，'
              f'产物 **{estimates["measured_artifact_bytes"] / 2**30:.2f} GiB**（含原始异常图，未含数据集和环境）。', "",
              "| 类别 | 方法（全量） | 候选向量 | 建库秒数 | 进程墙钟秒数 | CPU 峰值工作集 GiB | Torch 峰值分配 MiB | Torch 峰值预留 MiB | 推理中位 ms |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in resources:
        if r["sample_budget"] == "all":
            lines.append(f'| {r["category"]} | {r["method"]} | {r["candidate_vectors"]:,} | {r["feature_seconds"] + r["coreset_seconds"]:.2f} | '
                         f'{r["process_wall_seconds"]:.2f} | {r["peak_cpu_working_set_gib"]:.2f} | {r["peak_torch_allocated_mib"]:.1f} | '
                         f'{r["peak_torch_reserved_mib"]:.1f} | {r["inference_ms"]:.2f} |')
    lines += ["", f'所有运行保持记忆库 **1,000 × 384 × float32 = {1000 * 384 * 4 / 2**20:.3f} MiB**。'
              '本轮各档实际运行均完成，支持当前 224 分辨率、batch=1 配置在这台 8 GB 显卡上继续使用该容量。',
              'CPU 为每个独立进程至今峰值工作集；GPU 数字仅覆盖 PyTorch 分配器，不等于整卡占用，不包括驱动、桌面和其他应用。',
              '推理计时为 3 次预热后，固定 3 张开发图各重复 10 次的中位数；包含 CPU→GPU、特征、近邻、异常图及回传，排除磁盘/PIL。建库每配置只测一次，尚无运行间误差估计。', "",
              "## 三种子正式规模的成本预测（尚未执行）", "",
              f'按 A/B 72 次 + C 18 次，共 **{estimates["projected_formal_builds"]} 次建库**；'
              '最终评估图片数由已保存清单计数，不读取最终图像或分数。',
              f'时间点估计 **{estimates["projected_formal_seconds_point"] / 60:.1f} 分钟**；'
              f'增加 50% 调度余量后约 **{estimates["planning_seconds_with_50pct_buffer"] / 60:.1f} 分钟**。'
              f'输出文件估计约 **{estimates["projected_formal_artifact_bytes"] / 2**30:.1f} GiB**。',
              '估算将评估整阶段耗时按最终/开发图像数量放大；建库、校准和启动沿用实测。C 按两个视图在 A 一个/B 四个视图之间线性插值。排序复杂度、磁盘缓存、温度和未来代码变化会造成偏差；50% 是规划余量，不是统计置信区间。',
              '不含下载、人工复核、AU-PRO、另加方法、失败重跑及报告写作；协议尚未锁定，不能据此自动启动最终测试。', "",
              "## 可靠性与下一步", "",
              '- 本轮只有 seed=42，不计算跨种子标准差或显著性；小规模开发样本不足以确认 RQ1–RQ3。',
              '- 正常开发图 bottle/metal_nut 各 4 张、tile 7 张，FPR 每错一张分别跳动 25%/25%/14.29%，必须结合原始计数解释。',
              '- 完整保留所有 13 个条件与不利结果；重度扰动需结合接触表和裁剪比例审查，不能按最终得分删除条件。',
              '- 对照平滑/加权仍沿用已核对的简化基线；本轮仅增加计时采集，不改变模型、抽样或评分算法。',
              '- 下一步根据开发失败案例固定增强方案与压力测试说明，再进行三个种子的开发稳定性检查和 C 消融；正式评估需要先锁定协议。', "",
              "## 文件", "",
              '- `metrics.csv`：312 行类别/样本量/方法/条件指标。',
              '- `paired_differences.csv`：156 行相同样本和条件的 B−A。',
              '- `resources.csv`：24 组实测资源。',
              '- `formal_cost_projection.csv` / `resource_estimate.json`：成本推算与假设。',
              '- `clipping.csv`、`*_illumination.png`：开发扰动诊断，正常/异常掩码均按原图配对。',
              f'- `records/{plan["matrix"]["name"]}/`：不可变计划、核心源码快照、配置和摘要。',
              f'- `outputs/{plan["matrix"]["name"]}/`：原始逐图分数、异常图、模型、日志和恢复状态。', ""]
    (folder / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-name", default="dev_matrix_v1")
    summarize(parser.parse_args().matrix_name)
