"""Audit saved development scores and diagnose fixed-threshold failures; no inference."""
import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'data/cache/matplotlib'))
sys.path.insert(0, str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from anomaly.data import file_hash, save_json
from scripts.run_dev_matrix import verify_run

MATRIX = 'dev_matrix_v1'
OUT = ROOT / 'reports' / MATRIX / 'threshold_diagnostics'
STRESS = ['brightness_0.5', 'brightness_1.5', 'contrast_0.5', 'contrast_1.5']
BUDGETS = ['5', '10', '25', 'all']


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rows(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, values):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)


def close(actual, expected, context):
    if not np.isclose(actual, expected, rtol=0, atol=1e-12):
        raise ValueError(f'{context}: {actual} != {expected}')


def stats(values):
    return dict(zip(['min', 'q25', 'median', 'q75', 'q95', 'max'],
                    map(float, np.quantile(values, [0, .25, .5, .75, .95, 1], method='linear'))))


def main():
    plan_path = ROOT / 'records' / MATRIX / 'plan.json'
    status_path = ROOT / 'outputs' / MATRIX / 'status.json'
    plan, status = read(plan_path), read(status_path)
    assert status['status'] == 'complete'
    assert plan['matrix']['seeds'] == [42] and plan['matrix']['methods'] == ['A', 'B']
    inputs = {str(p.relative_to(ROOT)): file_hash(p) for p in [plan_path, status_path]}
    units, calibrations, conditions, cases = {}, [], [], []
    for job in plan['jobs']:
        entry, cfg = status['jobs'][job['id']], job['config']
        assert entry['status'] == 'complete' and cfg['evaluation_split'] == 'dev'
        folder = ROOT / entry['output']
        assert file_hash(folder / 'summary.json') == entry['summary_sha256']
        summary = verify_run(folder, cfg, plan['core_sha256'])
        samples = read(folder / 'sample_lists.json')
        cal, scored = rows(folder / 'calibration_scores.csv'), rows(folder / 'scores.csv')
        for name in ['config.json', 'summary.json', 'sample_lists.json', 'calibration_scores.csv', 'scores.csv', 'threshold.json']:
            p = folder / name
            inputs[p.relative_to(ROOT).as_posix()] = file_hash(p)
        assert len(cal) == len(samples['calibration'])
        assert {r['path'] for r in cal} == {r['path'] for r in samples['calibration']}
        calibration = np.array([float(r['score']) for r in cal])
        assert np.isfinite(calibration).all()
        threshold = float(np.quantile(calibration, cfg['threshold_quantile'], method='linear'))
        close(threshold, summary['threshold']['threshold'], job['id'])
        assert read(folder / 'threshold.json') == summary['threshold']
        close(float(np.mean(calibration > threshold)), summary['threshold']['calibration_fpr'], 'calibration FPR')
        meta = dict(category=cfg['category'], budget=str(cfg['normal_images']), method=cfg['method'], seed=cfg['seed'])
        calibrations.append({**meta, 'n': len(cal), 'threshold': threshold,
                             'false_positives': int((calibration > threshold).sum()),
                             'fpr': float(np.mean(calibration > threshold)), **stats(calibration)})
        expected = {r['path']: r for r in samples['dev']}
        metric_map = {r['condition']: r for r in summary['metrics']}
        score_map = {}
        for row in scored:
            label, score = int(row['label']), float(row['score'])
            assert label == expected[row['path']]['label'] and row['defect'] == expected[row['path']]['defect']
            assert np.isfinite(score) and int(row['predicted_anomaly']) == int(score > threshold)
            close(float(row['threshold']), threshold, 'constant threshold')
            score_map[(row['condition'], row['path'])] = score
        for condition in cfg['conditions']:
            name = condition['name']
            group = [r for r in scored if r['condition'] == name]
            normal = np.array([float(r['score']) for r in group if int(r['label']) == 0])
            abnormal = np.array([float(r['score']) for r in group if int(r['label']) == 1])
            fp, tp = int((normal > threshold).sum()), int((abnormal > threshold).sum())
            metric = metric_map[name]
            assert (fp, len(normal), tp, len(abnormal)) == tuple(metric[k] for k in ['false_positives', 'normal_images', 'true_positives', 'anomalous_images'])
            # Independent pairwise definition, rather than reusing the production rank implementation.
            auc = float(np.mean((abnormal[:, None] > normal) + .5 * (abnormal[:, None] == normal)))
            for actual, key in [(auc, 'image_auroc'), (fp / len(normal), 'fpr'), (tp / len(abnormal), 'recall')]:
                close(actual, metric[key], key)
            paired_shifts = []
            for row in group:
                clean_score = score_map[('clean', row['path'])]
                score = float(row['score'])
                if int(row['label']) == 0:
                    paired_shifts.append(score - clean_score)
                cases.append({**meta, 'condition': name, 'path': row['path'], 'label': int(row['label']),
                              'score': score, 'clean_score': clean_score, 'score_shift': score - clean_score,
                              'threshold': threshold, 'margin': score - threshold,
                              'predicted_anomaly': int(score > threshold)})
            conditions.append({**meta, 'condition': name, 'threshold': threshold,
                               'normal_n': len(normal), 'false_positives': fp, 'fpr': fp / len(normal),
                               'anomalous_n': len(abnormal), 'true_positives': tp, 'recall': tp / len(abnormal),
                               'image_auroc': auc, 'normal_median_score': float(np.median(normal)),
                               'normal_median_margin': float(np.median(normal - threshold)),
                               'normal_min_margin': float(min(normal - threshold)),
                               'normal_max_margin': float(max(normal - threshold)),
                               'normal_median_paired_shift': float(np.median(paired_shifts)),
                               'anomalous_min_margin': float(min(abnormal - threshold))})
        units[(meta['category'], meta['budget'], meta['method'])] = dict(
            threshold=threshold, calibration=calibration, scores=score_map, samples=samples,
            conditions=cfg['conditions'])
    swaps = []
    for category in plan['matrix']['categories']:
        for method in ['A', 'B']:
            small, full = units[(category, '5', method)], units[(category, 'all', method)]
            assert small['samples']['dev'] == full['samples']['dev']
            assert small['samples']['calibration'] == full['samples']['calibration']
            for condition in small['conditions']:
                name = condition['name']
                combo = {}
                for score_tag, model in [('5', small), ('all', full)]:
                    for threshold_tag, other in [('5', small), ('all', full)]:
                        for label, prefix in [(0, 'fp'), (1, 'tp')]:
                            values = [model['scores'][(name, r['path'])] for r in model['samples']['dev'] if r['label'] == label]
                            combo[f'{prefix}_scores{score_tag}_threshold{threshold_tag}'] = sum(v > other['threshold'] for v in values)
                swaps.append(dict(category=category, method=method, condition=name,
                                  threshold5=small['threshold'], thresholdall=full['threshold'],
                                  normal_n=sum(r['label'] == 0 for r in small['samples']['dev']),
                                  anomalous_n=sum(r['label'] == 1 for r in small['samples']['dev']), **combo))
    OUT.mkdir(parents=True, exist_ok=True)
    for name, values in [('calibration', calibrations), ('conditions', conditions), ('cases', cases), ('threshold_cross_checks', swaps)]:
        write_csv(OUT / f'{name}.csv', values)
    for category in plan['matrix']['categories']:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        colors = dict(zip(['clean', *STRESS], ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']))
        for row, method in enumerate(['A', 'B']):
            ax = axes[row, 0]
            ax.plot(BUDGETS, [units[(category, n, method)]['threshold'] for n in BUDGETS], 'ko-', label='Calibration threshold')
            for condition in ['clean', *STRESS]:
                values = [next(r for r in conditions if (r['category'], r['budget'], r['method'], r['condition']) == (category, n, method, condition)) for n in BUDGETS]
                ax.plot(BUDGETS, [r['normal_median_score'] for r in values], '.--', label=condition, color=colors[condition])
            ax.set(title=f'{method}: threshold and normal median', xlabel='Training originals', ylabel='Raw anomaly score')
            ax.legend(fontsize=7)
            for col, budget in [(1, '5'), (2, 'all')]:
                ax, model = axes[row, col], units[(category, budget, method)]
                series = [('Calibration clean', model['calibration'])]
                series += [(condition, [model['scores'][(condition, r['path'])] for r in model['samples']['dev'] if r['label'] == 0]) for condition in ['clean', *STRESS]]
                for label, values in series:
                    x = np.sort(values) / model['threshold']
                    ax.step(x, np.arange(1, len(x) + 1) / len(x), where='post', marker='.', color=colors.get(label, 'black'), label=f'{label} (n={len(x)})')
                ax.axvline(1, color='black', linestyle=':')
                ax.set(title=f'{method}: n={budget}, normal score ECDF', xlabel='Score / own threshold', ylabel='Cumulative fraction', ylim=(0, 1.05))
                ax.legend(fontsize=7, loc='lower left')
        fig.suptitle(f'{category} | development seed 42 | stress conditions shown; all conditions retained')
        fig.tight_layout(rect=(0, 0, 1, .96))
        fig.savefig(OUT / f'{category}.png', dpi=140)
        plt.close(fig)
    lines = ['# 固定阈值与光照误报诊断', '', '日期：2026-10-06。仅使用已有 seed=42 开发产物；不读取最终测试图像、不重新建库、不修改阈值。', '',
             f'已复核 {len(calibrations)} 个模型的校准分位数、{len(conditions)} 行条件指标及 {len(cases)} 条逐图分数。用独立两两比较重算图像 AUROC，并核对全部误报/检出计数。', '',
             '## 校准情况', '', '| 类别 | 正常校准图 | 超阈值图数 | 实际校准 FPR |', '|---|---:|---:|---:|']
    for category in plan['matrix']['categories']:
        selected = [r for r in calibrations if r['category'] == category]
        assert len({(r['n'], r['false_positives']) for r in selected}) == 1
        r = selected[0]
        lines.append(f"| {category} | {r['n']} | {r['false_positives']} | {r['fpr']:.2%} |")
    lines += ['', '各模型使用同一规则：干净正常校准分数的 95% 线性经验分位数，严格 score > threshold 判异常。有限校准样本下，实际校准 FPR 也未必恰好是 5%，更不能保证新图像的 FPR。', '',
              '## 全量模型的所有压力条件', '', '| 类别 | 方法 | 条件 | 正常误报 | 异常检出 | 图像 AUROC | 正常分数中位数 − 阈值 | 正常配对分数变化中位数 |', '|---|---|---|---:|---:|---:|---:|---:|']
    for r in conditions:
        if r['budget'] == 'all' and r['condition'] in STRESS:
            lines.append(f"| {r['category']} | {r['method']} | {r['condition']} | {r['false_positives']}/{r['normal_n']} | {r['true_positives']}/{r['anomalous_n']} | {r['image_auroc']:.4f} | {r['normal_median_margin']:+.4f} | {r['normal_median_paired_shift']:+.4f} |")
    lines += ['', '正 margin 表示正常图分数在阈值上方；配对变化是在同一模型内，用扰动分数减去同一原图的干净分数。AUROC 高并不意味着固定阈值下误报少。', '',
              '## 5 张到全量：阈值与分数共同变化', '',
              '下面展示 bottle 的两个已发现的压力案例。S5/Sall 为两种训练预算产生的分数，T5/Tall 为各自校准阈值。交叉使用阈值仅为数值诊断，既不是正式评测结果，也不是可部署阈值建议；不同模型的分数标度可能变化，不能据此作因果归因。', '',
              '| 方法 | 条件 | T5 → Tall | FP(S5,T5) | FP(S5,Tall) | FP(Sall,T5) | FP(Sall,Tall) | 全量模型异常检出 |', '|---|---|---|---:|---:|---:|---:|---:|']
    for r in swaps:
        if r['category'] == 'bottle' and r['condition'] in ['brightness_0.5', 'contrast_1.5']:
            counts = ' | '.join(f"{r[k]}/{r['normal_n']}" for k in ['fp_scores5_threshold5', 'fp_scores5_thresholdall', 'fp_scoresall_threshold5', 'fp_scoresall_thresholdall'])
            lines.append(f"| {r['method']} | {r['condition']} | {r['threshold5']:.6f} → {r['thresholdall']:.6f} | {counts} | {r['tp_scoresall_thresholdall']}/{r['anomalous_n']} |")
    lines += ['', '以 bottle A / brightness=0.5 为例：5 张时到全量时，阈值从 2.318011 降至 1.727753，正常扰动分数中位数同时上升，误报从 1/4 变为 4/4。这表明需要联合检查分数与阈值的相对位置，不能只凭阈值下降就解释全部变化。B 在同一亮度条件下降至 0/4 误报，但在 contrast=1.5 下仍为 4/4。', '', 'tile 全量 B 在 brightness=0.5 下误报从 A 的 2/7 降为 1/7，同时异常检出从 17/17 降为 16/17，说明误报改善也可能伴随漏检代价。本例仍只有一个种子。', '', '完整交叉检查覆盖三类、A/B 和全部 13 条件，同时保存异常检出计数。不要只看减少误报而忽略召回变化。', '',
              '## 范围与下一步', '',
              '- 63 张独立开发原图反复用于不同模型和条件；6,552 条评分不是 6,552 个独立样本。正常图每类仅 4、4、7 张。',
              '- 本次结果只描述已保存模型的分数和阈值关系，不证明样本数量的因果效应，也不支持显著性结论。',
              '- 保留 0.5/1.5 作为压力条件单独解读，所有 13 个条件均保留；延续训练增强 [0.8,1.2]、记忆库 1,000 和干净校准规则，不根据这几个误报案例重新调阈值。',
              '- 下一阶段补充 seed=17、2026 的开发 A/B，并在 seed=42、17、2026 下做 5、25 张的 C 消融。种子预先指定，不按结果挑选。完成后再锁定正式协议。', '',
              '## 文件和复跑', '',
              '- calibration.csv：24 组校准分布与实际 FPR。',
              '- conditions.csv：312 行全部条件的分数、margin、误报和召回。',
              '- cases.csv：6,552 条同原图配对分数与 margin。',
              '- threshold_cross_checks.csv：78 组四种分数/阈值组合的 FP 和 TP。',
              '- bottle.png / metal_nut.png / tile.png：阈值随预算变化及正常分数经验分布。',
              '- provenance.json：分析脚本与所有输入文件的 SHA-256，记录未评估最终测试。', '',
              '```powershell', r'Set-Location D:\ChatGPT\111', r'.\.venv-audit\Scripts\python.exe scripts/diagnose_dev_thresholds.py', '```', '']
    (OUT / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    save_json(OUT / 'provenance.json', dict(matrix=MATRIX, final_test_evaluated=False,
        models=len(calibrations), condition_rows=len(conditions), score_rows=len(cases),
        script_sha256=file_hash(Path(__file__)), inputs_sha256=inputs,
        numpy=np.__version__, matplotlib=matplotlib.__version__))
    print(json.dumps(dict(output=str(OUT), models=len(calibrations), conditions=len(conditions), scores=len(cases)), ensure_ascii=False))


if __name__ == '__main__':
    main()
