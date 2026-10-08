"""Validate all 90 development models and aggregate paired three-seed results."""
import csv
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'data/cache/matplotlib'))
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from anomaly.data import file_hash, save_json, training_views
from scripts.run_dev_matrix import verify_run
from scripts.development_statistics import (seed_summary, paired_differences,
                                            macro_categories, condition_groups)

MATRICES = ['dev_matrix_v1', 'dev_ab_extra_seeds_v1', 'dev_c_ablation_v1']
CATEGORIES = ['bottle', 'metal_nut', 'tile']
SEEDS = [17, 42, 2026]
BUDGETS = ['5', '10', '25', 'all']
METRICS = ['image_auroc', 'pixel_auroc', 'fpr', 'recall', 'image_auroc_drop', 'fpr_increase']
OUT = ROOT / 'reports/dev_stability_v1'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def read_csv(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(a, b, message):
    require(bool(np.isclose(a, b, rtol=0, atol=1e-12)), message)


def check_scores(folder, cfg, summary, samples):
    calibration = read_csv(folder / 'calibration_scores.csv')
    require(len(calibration) == len(samples['calibration']) and
            {r['path'] for r in calibration} == {r['path'] for r in samples['calibration']}, 'Calibration sample mismatch')
    values = np.asarray([float(r['score']) for r in calibration])
    require(np.isfinite(values).all(), 'Nonfinite calibration scores')
    threshold = float(np.quantile(values, cfg['threshold_quantile'], method='linear'))
    close(threshold, summary['threshold']['threshold'], 'Calibration threshold mismatch')
    close(float(np.mean(values > threshold)), summary['threshold']['calibration_fpr'], 'Calibration FPR mismatch')
    require(read(folder / 'threshold.json') == summary['threshold'], 'Threshold record mismatch')
    samples_by_path = {r['path']: r for r in samples['dev']}
    scored = read_csv(folder / 'scores.csv')
    groups = defaultdict(list)
    for row in scored:
        label, score = int(row['label']), float(row['score'])
        require(np.isfinite(score), 'Nonfinite evaluation score')
        require(label == samples_by_path[row['path']]['label'], 'Label mismatch')
        require(int(row['predicted_anomaly']) == int(score > threshold), 'Decision mismatch')
        groups[row['condition']].append((label, score))
    clean = next(m for m in summary['metrics'] if m['condition'] == 'clean')
    for metric in summary['metrics']:
        rows = groups[metric['condition']]
        normal = np.array([s for label, s in rows if label == 0])
        abnormal = np.array([s for label, s in rows if label == 1])
        fp, tp = int(sum(normal > threshold)), int(sum(abnormal > threshold))
        require((fp, len(normal), tp, len(abnormal)) == tuple(metric[k] for k in
            ['false_positives', 'normal_images', 'true_positives', 'anomalous_images']), 'Confusion count mismatch')
        independent_auc = np.mean((abnormal[:, None] > normal) + .5 * (abnormal[:, None] == normal))
        for actual, key in [(independent_auc, 'image_auroc'), (fp / len(normal), 'fpr'), (tp / len(abnormal), 'recall'),
                            (clean['image_auroc'] - metric['image_auroc'], 'image_auroc_drop'),
                            (metric['fpr'] - clean['fpr'], 'fpr_increase')]:
            close(actual, metric[key], key + ' mismatch')
        require(0 <= metric['pixel_auroc'] <= 1, 'Invalid pixel AUROC')
    return len(scored)


def collect():
    reference = read(ROOT / 'records/dev_matrix_v1/plan.json')
    follow = read(ROOT / 'records/dev_followup_plan.json')
    for item in follow['plans']:
        require(file_hash(ROOT / item['config']) == item['sha256'], 'Predeclared matrix changed')
    conditions = reference['matrix']['conditions']
    common = {k: v for k, v in reference['jobs'][0]['config'].items()
              if k not in {'category', 'normal_images', 'method', 'seed'}}
    inputs, units, metrics, resources = {}, {}, [], []
    score_count = 0
    for name in MATRICES:
        record = ROOT / 'records' / name
        plan, status = read(record / 'plan.json'), read(ROOT / 'outputs' / name / 'status.json')
        require(status.get('status') == 'complete', f'Incomplete matrix: {name}')
        require(set(status['jobs']) == {j['id'] for j in plan['jobs']}, 'Unexpected job IDs')
        for field in ['core_sha256', 'manifest_sha256', 'weights_sha256']:
            require(plan[field] == reference[field], 'Incompatible matrices: ' + field)
        for path in [record / 'plan.json', ROOT / 'outputs' / name / 'status.json']:
            inputs[path.relative_to(ROOT).as_posix()] = file_hash(path)
        for job in plan['jobs']:
            cfg, entry = job['config'], status['jobs'][job['id']]
            require(entry['status'] == 'complete', 'Incomplete run')
            require({k: v for k, v in cfg.items() if k in common} == common, 'Common configuration mismatch')
            folder = ROOT / entry['output']
            require(file_hash(folder / 'summary.json') == entry['summary_sha256'], 'Changed summary')
            summary = verify_run(folder, cfg, reference['core_sha256'])
            require(read(folder / 'provenance.json')['weights_sha256'] == reference['weights_sha256'], 'Weights mismatch')
            samples, views = read(folder / 'sample_lists.json'), read(folder / 'training_views.json')
            expected_views = [meta for row in samples['train'] for _, meta in
                              training_views(torch.zeros(3, 1, 1), row, cfg['seed'], {'A': 0, 'B': 3, 'C': 1}[cfg['method']])]
            require(views == expected_views, 'Training augmentation mismatch')
            require(summary['training_originals'] == len(samples['train']), 'Training count mismatch')
            require(summary['augmented_copies'] == len(views) - len(samples['train']), 'View count mismatch')
            require(summary['candidate_vectors'] == len(views) * 784, 'Candidate count differs from 224-input features')
            model = torch.load(folder / 'model.pt', map_location='cpu', weights_only=True)
            bank, indices = model['memory_bank'], model['selected_indices']
            require(tuple(bank.shape) == (1000, 384) and bank.dtype == torch.float32, 'Saved bank shape/dtype mismatch')
            require(bool(torch.isfinite(bank).all()), 'Nonfinite saved bank')
            require(tuple(indices.shape) == (1000,) and len(indices.unique()) == 1000, 'Invalid coreset indices')
            require(int(indices.min()) >= 0 and int(indices.max()) < summary['candidate_vectors'], 'Coreset index out of range')
            require(model['config'] == cfg and model['threshold'] == summary['threshold']['threshold'], 'Saved model configuration mismatch')
            require(model['manifest_sha256'] == reference['manifest_sha256'] and model['weights_sha256'] == reference['weights_sha256'], 'Saved model provenance mismatch')
            require(summary['memory_bytes'] == bank.numel() * bank.element_size(), 'Bank byte count mismatch')
            del model, bank, indices
            score_count += check_scores(folder, cfg, summary, samples)
            meta = dict(category=cfg['category'], budget=str(cfg['normal_images']), method=cfg['method'], seed=cfg['seed'])
            key = (meta['category'], meta['budget'], meta['method'], meta['seed'])
            require(key not in units, 'Duplicate run')
            units[key] = dict(summary=summary, samples=samples, views=views)
            metrics.extend({**meta, **m} for m in summary['metrics'])
            phases = summary['resources']
            resources.append({**meta, 'matrix': name, 'training_originals': summary['training_originals'],
                'augmented_copies': summary['augmented_copies'], 'calibration_images': summary['calibration_images'],
                'dev_images': summary['dev_images'], 'candidate_vectors': summary['candidate_vectors'],
                'bank_vectors': summary['memory_vectors'], 'bank_bytes': summary['memory_bytes'],
                'build_seconds': phases['feature_extraction']['seconds'] + phases['coreset']['seconds'],
                'evaluation_seconds': phases['evaluation']['seconds'], 'process_wall_seconds': entry['process_wall_seconds'],
                'inference_ms': 1000 * summary['repeated_inference_median_seconds'], 'artifact_bytes': entry['artifact_bytes'],
                'peak_cpu_gib': max(p.get('process_peak_working_set_bytes_so_far') or 0 for p in phases.values()) / 2**30,
                'peak_torch_allocated_mib': max(p['cuda_peak_allocated_bytes'] or 0 for p in phases.values()) / 2**20,
                'peak_torch_reserved_mib': max(p['cuda_peak_reserved_bytes'] or 0 for p in phases.values()) / 2**20})
            for filename in ['config.json', 'provenance.json', 'summary.json', 'sample_lists.json', 'training_views.json',
                             'scores.csv', 'calibration_scores.csv', 'threshold.json', 'model.pt']:
                path = folder / filename
                inputs[path.relative_to(ROOT).as_posix()] = file_hash(path)
        print(f'Validated {name}: {len(plan["jobs"])} models', flush=True)
    expected = {(c,n,m,s) for c in CATEGORIES for n in BUDGETS for s in SEEDS
                for m in (['A','B','C'] if n in ['5','25'] else ['A','B'])}
    require(set(units) == expected, 'Incomplete combined design')
    for category in CATEGORIES:
        eval_reference = units[(category, '5', 'A', 42)]['samples']
        for seed in SEEDS:
            full = units[(category, 'all', 'A', seed)]['samples']['train']
            require({r['path'] for r in full} == {r['path'] for r in units[(category,'all','A',42)]['samples']['train']}, 'Full pools differ')
            for budget in BUDGETS:
                a, b = [units[(category,budget,m,seed)] for m in ['A','B']]
                require(a['samples'] == b['samples'], 'A/B sample mismatch')
                for split in ['dev', 'calibration']:
                    require(a['samples'][split] == eval_reference[split], 'Evaluation/calibration mismatch across seeds')
                train = a['samples']['train']
                require(train == full[:len(train)], 'Non-nested training samples')
                if budget in ['5','25']:
                    c = units[(category,budget,'C',seed)]
                    require(c['samples'] == a['samples'], 'A/C sample mismatch')
                    require(c['views'] == [v for v in b['views'] if v['view'] <= 1], 'C views not paired with B')
    manifest = read(ROOT / reference['jobs'][0]['config']['manifest'])
    counts = {c: {split: sum(r['category'] == c and r['split'] == split for r in manifest['rows'])
                  for split in ['dev', 'calibration', 'train_pool', 'final_test']} for c in CATEGORIES}
    return metrics, resources, units, conditions, inputs, counts, score_count


def main():
    metrics, resources, units, conditions, inputs, counts, score_count = collect()
    OUT.mkdir(parents=True, exist_ok=True)
    pairs = paired_differences(metrics, METRICS)
    group = ['category','budget','method','condition']
    pair_group = ['category','budget','comparison','condition']
    tables = {'metrics': metrics, 'resources': resources, 'paired_by_seed': pairs,
              'metrics_seed_summary': seed_summary(metrics, group, METRICS),
              'paired_seed_summary': seed_summary(pairs, pair_group, METRICS)}
    macro = macro_categories(metrics, ['budget','method','seed','condition'], METRICS)
    macro_pairs = macro_categories(pairs, ['budget','comparison','seed','condition'], METRICS)
    tables['macro_by_seed'] = macro
    tables['macro_paired_by_seed'] = macro_pairs
    tables['macro_seed_summary'] = seed_summary(macro, ['budget','method','condition'], METRICS)
    tables['macro_paired_seed_summary'] = seed_summary(macro_pairs, ['budget','comparison','condition'], METRICS)
    severity = condition_groups(metrics, ['category','budget','method','seed'], METRICS, conditions)
    severity_pairs = condition_groups(pairs, ['category','budget','comparison','seed'], METRICS, conditions)
    tables['severity_by_seed'] = severity
    tables['severity_paired_by_seed'] = severity_pairs
    tables['severity_seed_summary'] = seed_summary(severity, ['category','budget','method','severity'], METRICS)
    tables['severity_paired_seed_summary'] = seed_summary(severity_pairs, ['category','budget','comparison','severity'], METRICS)
    macro_severity = macro_categories(severity, ['budget','method','seed','severity'], METRICS)
    macro_severity_pairs = macro_categories(severity_pairs, ['budget','comparison','seed','severity'], METRICS)
    tables['macro_severity_by_seed'] = macro_severity
    tables['macro_severity_paired_by_seed'] = macro_severity_pairs
    tables['macro_severity_seed_summary'] = seed_summary(macro_severity, ['budget','method','severity'], METRICS)
    tables['macro_severity_paired_seed_summary'] = seed_summary(macro_severity_pairs, ['budget','comparison','severity'], METRICS)
    tables['resources_seed_summary'] = seed_summary(resources, ['category','budget','method'],
        ['build_seconds','process_wall_seconds','inference_ms','candidate_vectors','peak_cpu_gib'])
    for name, rows in tables.items():
        write_csv(OUT / (name + '.csv'), rows)
    monotonic = []
    for category in CATEGORIES:
        for seed in SEEDS:
            for method in ['A','B']:
                for condition in conditions:
                    if condition['name'] == 'clean':
                        continue
                    sequence = [next(r for r in metrics if (r['category'],r['seed'],r['method'],r['budget'],r['condition']) ==
                                     (category,seed,method,budget,condition['name'])) for budget in BUDGETS]
                    row = dict(category=category,seed=seed,method=method,condition=condition['name'])
                    for metric in ['image_auroc_drop','fpr_increase']:
                        values = [r[metric] for r in sequence]
                        differences = np.diff(values)
                        if np.all(np.abs(differences) <= 1e-12):
                            pattern = 'all_tied'
                        elif np.all(differences <= 1e-12):
                            pattern = 'nonincreasing'
                        elif np.all(differences >= -1e-12):
                            pattern = 'nondecreasing'
                        else:
                            pattern = 'nonmonotone'
                        row[metric + '_pattern'] = pattern
                        row.update({metric + '_' + budget: value for budget,value in zip(BUDGETS,values)})
                    monotonic.append(row)
    tables['rq2_monotonicity'] = monotonic
    write_csv(OUT / 'rq2_monotonicity.csv', monotonic)
    projected_seconds = sum(r['process_wall_seconds'] + r['evaluation_seconds'] *
                            (counts[r['category']]['final_test']/counts[r['category']]['dev'] - 1) for r in resources)
    projected_bytes = sum(r['artifact_bytes'] * counts[r['category']]['final_test']/counts[r['category']]['dev'] for r in resources)
    overview = dict(status='development_only', model_count=len(units), new_model_count=66, seed_count=3,
        condition_rows=len(metrics), scored_views=score_count, independent_dev_images=sum(c['dev'] for c in counts.values()),
        final_test_evaluated=False, seeds=SEEDS, counts=counts,
        new_process_wall_seconds=sum(r['process_wall_seconds'] for r in resources if r['matrix']!='dev_matrix_v1'),
        all_process_wall_seconds=sum(r['process_wall_seconds'] for r in resources),
        new_artifact_bytes=sum(r['artifact_bytes'] for r in resources if r['matrix']!='dev_matrix_v1'),
        all_artifact_bytes=sum(r['artifact_bytes'] for r in resources),
        max_cpu_gib=max(r['peak_cpu_gib'] for r in resources),
        max_torch_allocated_mib=max(r['peak_torch_allocated_mib'] for r in resources),
        max_torch_reserved_mib=max(r['peak_torch_reserved_mib'] for r in resources),
        formal_projected_seconds=projected_seconds, formal_planning_seconds=projected_seconds*1.5,
        formal_projected_bytes=projected_bytes,
        projection_note='Uses actual A/B/C build timings; scales evaluation by final/dev image count; heuristic 50% time buffer; excludes AU-PRO, manual analysis and reruns.',
        checks=['Same core, manifest, weights and fixed configuration across matrices',
                '90 expected jobs, nested training budgets, A/B/C samples and B/C augmentation pairing',
                '1000-vector capacity; all 13 conditions; independent image AUROC and confusion-count checks',
                'Calibration thresholds reconstructed; final evaluation absent',
                'Per-category and macro statistics computed across three seeds with ddof=1'])
    save_json(OUT / 'overview.json', overview)
    sources={p.relative_to(ROOT).as_posix():file_hash(p) for p in
             [Path(__file__), ROOT/'scripts/development_statistics.py', ROOT/'scripts/render_dev_stability.py']}
    save_json(OUT / 'provenance.json', dict(inputs_sha256=inputs, analysis_source_sha256=sources,
                                         final_test_evaluated=False, numpy=np.__version__))
    snapshot = ROOT / 'records/dev_stability_v1/source'
    for relative in sources:
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    from scripts.render_dev_stability import render
    render(OUT, tables, overview)
    save_json(ROOT / 'records/dev_stability_v1/completion.json', dict(
        status='complete', final_test_evaluated=False, report=OUT.relative_to(ROOT).as_posix(),
        model_count=len(units), new_model_count=66, sources_sha256=sources,
        matrix_status_sha256={name: file_hash(ROOT/'outputs'/name/'status.json') for name in MATRICES}))
    print(json.dumps(overview, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
