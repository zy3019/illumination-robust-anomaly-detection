"""Aggregate paired development experiments without treating conditions as replicates."""
from collections import defaultdict
import numpy as np


def seed_summary(rows, group_keys, metrics, seeds=(17, 42, 2026)):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in group_keys)].append(row)
    output = []
    for key, group in sorted(groups.items()):
        found = [r['seed'] for r in group]
        if len(found) != len(seeds) or set(found) != set(seeds):
            raise ValueError(f'Expected exactly one row per seed: {key}: {found}')
        result = dict(zip(group_keys, key), seed_count=len(seeds))
        for metric in metrics:
            values = np.asarray([r[metric] for r in group], dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f'Nonfinite metric: {key}/{metric}')
            result[metric + '_mean'] = float(values.mean())
            result[metric + '_std'] = float(values.std(ddof=1))
        output.append(result)
    return output


def paired_differences(rows, metrics):
    index = {}
    keys = ['category', 'budget', 'seed', 'condition']
    for row in rows:
        key = tuple(row[k] for k in keys)
        group = index.setdefault(key, {})
        if row['method'] in group:
            raise ValueError(f'Duplicate method: {key}')
        group[row['method']] = row
    output = []
    for key, group in sorted(index.items()):
        expected = {'A', 'B', 'C'} if key[1] in ('5', '25') else {'A', 'B'}
        if set(group) != expected:
            raise ValueError(f'Missing or extra paired method: {key}')
        comparisons = [('B', 'A')] + ([('C', 'A'), ('B', 'C')] if 'C' in group else [])
        for left, right in comparisons:
            output.append({**dict(zip(keys, key)), 'comparison': f'{left}-{right}',
                           **{m: group[left][m] - group[right][m] for m in metrics}})
    return output


def macro_categories(rows, group_keys, metrics, categories=('bottle', 'metal_nut', 'tile')):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in group_keys)].append(row)
    output = []
    for key, group in sorted(groups.items()):
        found = [r['category'] for r in group]
        if len(found) != len(categories) or set(found) != set(categories):
            raise ValueError(f'Incomplete category macro: {key}: {found}')
        output.append({**dict(zip(group_keys, key)),
                       **{m: float(np.mean([r[m] for r in group])) for m in metrics}})
    return output


def condition_groups(rows, group_keys, metrics, conditions):
    names = {c['name'] for c in conditions}
    subsets = {
        'clean': {'clean'},
        'mild': {c['name'] for c in conditions if c['operation'] != 'clean' and c['factor'] in (.9, 1.1)},
        'moderate': {c['name'] for c in conditions if c['factor'] in (.7, 1.3)},
        'stress': {c['name'] for c in conditions if c['factor'] in (.5, 1.5)},
        'all_perturbed': names - {'clean'},
    }
    if any(not s for s in subsets.values()):
        raise ValueError('Missing severity conditions')
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in group_keys)].append(row)
    output = []
    for key, group in sorted(groups.items()):
        if len(group) != len(names) or {r['condition'] for r in group} != names:
            raise ValueError(f'Incomplete conditions: {key}')
        for severity, subset in subsets.items():
            selected = [r for r in group if r['condition'] in subset]
            output.append({**dict(zip(group_keys, key)), 'severity': severity,
                           **{m: float(np.mean([r[m] for r in selected])) for m in metrics}})
    return output
