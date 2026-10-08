"""Validate a paired A/B development run and write a reproducible summary."""
import argparse
import csv
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="outputs/bottle_A_seed42_n5")
    parser.add_argument("--augmented", default="outputs/bottle_B_seed42_n5")
    parser.add_argument("--output", default="reports/bottle_pilot.md")
    args = parser.parse_args()
    a_path, b_path = Path(args.baseline), Path(args.augmented)
    a, b = read(a_path / "summary.json"), read(b_path / "summary.json")
    if (a["method"], b["method"]) != ("A", "B"):
        raise ValueError("Expected A then B")
    for key in ["category", "seed", "training_originals", "memory_vectors", "memory_bytes", "calibration_images", "dev_images"]:
        if a[key] != b[key]:
            raise ValueError(f"Unpaired comparison: {key}")
    if read(a_path / "sample_lists.json") != read(b_path / "sample_lists.json"):
        raise ValueError("Sample lists differ")
    ac, bc = read(a_path / "config.json"), read(b_path / "config.json")
    ac.pop("method")
    bc.pop("method")
    if ac != bc:
        raise ValueError("Configurations differ beyond method")
    ap, bp = read(a_path / "provenance.json"), read(b_path / "provenance.json")
    for key in ["manifest_sha256", "weights_sha256", "source_sha256"]:
        if ap[key] != bp[key]:
            raise ValueError(f"Provenance differs: {key}")
    baseline_metrics = {r["condition"]: r for r in a["metrics"]}
    augmented_metrics = {r["condition"]: r for r in b["metrics"]}
    if baseline_metrics.keys() != augmented_metrics.keys():
        raise ValueError("Conditions differ")
    for path, summary in [(a_path, a), (b_path, b)]:
        with (path / "scores.csv").open(encoding="utf-8") as stream:
            score_rows = list(csv.DictReader(stream))
        expected = summary["dev_images"] * len(summary["metrics"])
        if len(score_rows) != expected or len({(r["condition"], r["path"]) for r in score_rows}) != expected:
            raise ValueError("Missing or duplicate per-image scores")
        if any(float(r["threshold"]) != summary["threshold"]["threshold"] for r in score_rows):
            raise ValueError("Threshold changed between conditions")
        if len(list((path / "maps").rglob("*.npy"))) != expected:
            raise ValueError("Missing anomaly maps")
    clean = baseline_metrics["clean"]
    normal_count, anomaly_count = clean["normal_images"], clean["anomalous_images"]
    equal_image_metrics = all(
        baseline_metrics[name][key] == augmented_metrics[name][key]
        for name in baseline_metrics for key in ["image_auroc", "fpr", "recall"]
    )
    image_observation = (
        '这组小规模开发试验未观察到图像 AUROC、误报或召回上的增强收益。'
        if equal_image_metrics else
        '两组图像指标存在差异；需结合上表、更多种子与类别评价收益和代价。'
    )
    lines = [
        f'# {a["category"]} 首轮开发试验', "",
        f'运行时间：{ap["started_at"]}；状态：开发试运行，未评估最终测试集。', "",
        f'使用 {a["training_originals"]} 张独立正常训练图、{a["calibration_images"]} 张额外正常校准图、'
        f'{a["dev_images"]} 张开发图（{normal_count} 正常 / {anomaly_count} 异常），随机种子 {a["seed"]}。', "",
        'A 为原图基线；B 每张原图增加 3 个光照增强视图。两组样本清单、骨干权重、源代码、配置（除方法）及最终记忆库容量均已核验一致。', "",
        f'记忆库各 {a["memory_vectors"]:,} 个 384 维向量，{a["memory_bytes"]:,} 字节；'
        f'A 候选 {a["candidate_vectors"]:,} 个，B 候选 {b["candidate_vectors"]:,} 个。', "",
        '| 条件 | A 图像 AUROC | B 图像 AUROC | A 像素 AUROC | B 像素 AUROC | B−A 像素 AUROC |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for name, am in baseline_metrics.items():
        bm = augmented_metrics[name]
        lines.append(f'| {name} | {am["image_auroc"]:.4f} | {bm["image_auroc"]:.4f} | '
                     f'{am["pixel_auroc"]:.4f} | {bm["pixel_auroc"]:.4f} | {bm["pixel_auroc"] - am["pixel_auroc"]:+.6f} |')
    lines += ["", "## 固定阈值结果", "",
              '| 方法 | 条件 | 误报 / 正常图 | 检出 / 异常图 |', '|---|---|---:|---:|']
    for summary in [a, b]:
        for result in summary["metrics"]:
            lines.append(f'| {summary["method"]} | {result["condition"]} | {result["false_positives"]}/{result["normal_images"]} | '
                         f'{result["true_positives"]}/{result["anomalous_images"]} |')
    lines += ["", f'A 阈值 {a["threshold"]["threshold"]:.6f}；B 阈值 {b["threshold"]["threshold"]:.6f}。'
              '各自只在干净正常校准集上确定，并在全部条件固定。', "",
              '## 当前观察与限制', "",
              f'- {image_observation}像素 AUROC 的差异不作显著性解释。',
              f'- A 干净图像 AUROC={clean["image_auroc"]:.4f}，正常图误报 {clean["false_positives"]}/{normal_count}；排序指标与阈值表现需分别考察。',
              f'- 正常开发图仅 {normal_count} 张，误报率每一张变化 {100 / normal_count:.2f}%；当前不足以对误报率作稳定估计。',
              f'- 只跑了 1 类、1 个样本数量、1 个种子和 {len(baseline_metrics)} 个条件，不能回答完整 RQ1–RQ3。',
              '- 计时仅为管线诊断。A 首次运行与 B 后续运行可能受设备预热/缓存影响，不据此声称增强加快建库。',
              '- 定位质量需结合开发图片、原始异常图和缺陷掩码检查，不能只凭像素 AUROC 下结论。', "",
              '## 原始记录', "", f'- A：`{a_path.as_posix()}`', f'- B：`{b_path.as_posix()}`',
              f'- 每组已核验 {expected} 条逐图评分和 {expected} 个原始异常图完整；热力图 PNG 每条件每种缺陷保留一张。',
              '- 运行配置：`configs/bottle_pilot.json`；实现边界：`docs/implementation.md`。', "",
              '下一步：核对框架实现差异，扩展三类的开发试验与资源估算，再锁定正式实验协议。', ""]
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines), encoding="utf-8")
    print(f"Paired artifacts verified; report: {target}")


if __name__ == "__main__":
    main()
