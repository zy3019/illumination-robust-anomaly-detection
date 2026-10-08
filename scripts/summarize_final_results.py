"""Audit and summarize the complete frozen final matrix; no method selection."""
import argparse
import csv
import os
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'data/cache/matplotlib'))
sys.path.insert(0,str(ROOT))
import numpy as np
from anomaly.data import file_hash,save_json,load_mask
from anomaly.metrics import auroc
from scripts.final_protocol import read,require,load_locked
from scripts.run_final_matrix import verify_result
from scripts.development_statistics import seed_summary,paired_differences,macro_categories,condition_groups
from scripts.summarize_dev_stability import read_csv,write_csv,close,METRICS,CATEGORIES,SEEDS,BUDGETS


def main():
    global ROOT
    parser=argparse.ArgumentParser()
    parser.add_argument('--protocol',default='records/formal_v1/protocol.json')
    parser.add_argument('--digest',required=True)
    parser.add_argument('--frozen-root', type=Path)
    args=parser.parse_args()
    if args.frozen_root:
        ROOT = args.frozen_root.resolve()
    import scripts.run_final_matrix as runner
    runner.ROOT = ROOT
    protocol,_=load_locked(ROOT/args.protocol,args.digest,root=ROOT)
    OUT=ROOT/'reports'/protocol['id']
    status_path=ROOT/'outputs'/protocol['id']/'status.json'
    status=read(status_path)
    require(status.get('status')=='complete' and len(status['jobs'])==90,'Incomplete final matrix')
    require(status['protocol_sha256']==args.digest,'Status protocol mismatch')
    conditions=protocol['conditions'];metrics=[];resources=[];audits=[]
    inputs={args.protocol:args.digest,status_path.relative_to(ROOT).as_posix():file_hash(status_path)}
    for job in protocol['jobs']:
        entry=status['jobs'][job['id']];cfg=job['config'];folder=ROOT/entry['output']
        require(entry['status']=='complete','Incomplete job')
        summary=verify_result(folder,job,protocol,args.digest,entry['integrity_sha256'])
        samples=protocol['final_samples'][cfg['category']]
        sample_by_path={r['path']:r for r in samples}
        scores=read_csv(folder/'scores.csv')
        for r in scores:
            require(int(r['label'])==sample_by_path[r['path']]['label'],'Score label mismatch')
            require(np.isfinite(float(r['score'])) and int(r['predicted_anomaly'])==int(float(r['score'])>float(r['threshold'])),'Score decision mismatch')
        clean=next(m for m in summary['metrics'] if m['condition']=='clean')
        for m in summary['metrics']:
            normal=np.array([float(r['score']) for r in scores if r['condition']==m['condition'] and r['label']=='0'])
            abnormal=np.array([float(r['score']) for r in scores if r['condition']==m['condition'] and r['label']=='1'])
            threshold=summary['threshold']['threshold']
            fp,tp=int(sum(normal>threshold)),int(sum(abnormal>threshold))
            require((fp,len(normal),tp,len(abnormal))==tuple(m[k] for k in ['false_positives','normal_images','true_positives','anomalous_images']),'Confusion count mismatch')
            for actual,key in [(np.mean((abnormal[:,None]>normal)+.5*(abnormal[:,None]==normal)),'image_auroc'),
                                (fp/len(normal),'fpr'),(tp/len(abnormal),'recall'),
                                (clean['image_auroc']-m['image_auroc'],'image_auroc_drop'),(m['fpr']-clean['fpr'],'fpr_increase')]:
                close(actual,m[key],key+' audit')
            require(0<=m['pixel_auroc']<=1,'Invalid pixel metric')
        if job['id'] in protocol['pixel_audit_jobs']:
            labels=np.concatenate([load_mask(ROOT/cfg['data_root'],r,cfg['image_size']).ravel() for r in samples])
            for condition in protocol['pixel_audit_conditions']:
                maps=np.concatenate([np.load(folder/'maps'/condition/f"{r['defect']}_{Path(r['path']).stem}.npy").ravel() for r in samples])
                reference=auroc(labels,maps)
                saved=next(m['pixel_auroc'] for m in summary['metrics'] if m['condition']==condition)
                close(reference,saved,'Independent pixel AUROC audit')
                audits.append(dict(job=job['id'],condition=condition,reference=reference,saved=saved,absolute_difference=abs(reference-saved)))
        meta=dict(category=cfg['category'],budget=str(cfg['normal_images']),method=cfg['method'],seed=cfg['seed'])
        metrics.extend({**meta,**m} for m in summary['metrics'])
        build=read(ROOT/job['source_output']/'summary.json');phases=summary['resources']
        resources.append({**meta,'training_originals':build['training_originals'],'candidate_vectors':build['candidate_vectors'],
            'build_seconds':build['resources']['feature_extraction']['seconds']+build['resources']['coreset']['seconds'],
            'build_cost_source':job['source_output'],'process_wall_seconds':entry['process_wall_seconds'],
            'inference_ms':summary['repeated_inference_median_seconds']*1000,'artifact_bytes':entry['artifact_bytes'],
            'peak_cpu_gib':max(p.get('process_peak_working_set_bytes_so_far') or 0 for p in phases.values())/2**30,
            'peak_torch_allocated_mib':max(p['cuda_peak_allocated_bytes'] or 0 for p in phases.values())/2**20,
            'peak_torch_reserved_mib':max(p['cuda_peak_reserved_bytes'] or 0 for p in phases.values())/2**20})
        inputs[(folder/'integrity.json').relative_to(ROOT).as_posix()]=entry['integrity_sha256']
        print('Audited '+job['id'],flush=True)
    require(len(metrics)==1170 and len(audits)==6,'Unexpected summary size')
    OUT.mkdir(parents=True,exist_ok=True)
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
    resource_metrics = ['build_seconds','inference_ms','candidate_vectors','peak_cpu_gib']
    if all(r['process_wall_seconds'] is not None for r in resources):
        resource_metrics.append('process_wall_seconds')
    tables['resources_seed_summary'] = seed_summary(resources, ['category','budget','method'], resource_metrics)
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
    manifest=read(ROOT/protocol['manifest'])
    counts={c:dict(normal=sum(r['label']==0 for r in protocol['final_samples'][c]),
                   anomalous=sum(r['label']==1 for r in protocol['final_samples'][c]),
                   calibration=sum(r['category']==c and r['split']=='calibration' for r in manifest['rows']),
                   train_pool=sum(r['category']==c and r['split']=='train_pool' for r in manifest['rows'])) for c in CATEGORIES}
    overview=dict(status='final_test_complete',protocol_sha256=args.digest,model_count=90,
                  score_rows=sum(len(protocol['final_samples'][j['config']['category']])*13 for j in protocol['jobs']),
                  independent_final_images=sum(len(r) for r in protocol['final_samples'].values()),counts=counts,
                  process_wall_seconds=sum(r['process_wall_seconds'] for r in resources if r['process_wall_seconds'] is not None),
                  missing_process_wall_times=sum(r['process_wall_seconds'] is None for r in resources),
                  artifact_bytes=sum(r['artifact_bytes'] for r in resources),
                  peak_cpu_gib=max(r['peak_cpu_gib'] for r in resources),
                  peak_torch_allocated_mib=max(r['peak_torch_allocated_mib'] for r in resources),
                  peak_torch_reserved_mib=max(r['peak_torch_reserved_mib'] for r in resources),
                  pixel_audit_count=len(audits),final_test_evaluated=True)
    save_json(OUT/'overview.json',overview);save_json(OUT/'pixel_audit.json',audits)
    save_json(OUT/'provenance.json',dict(protocol_sha256=args.digest,inputs_sha256=inputs,locked_source_sha256=protocol['source_sha256']))
    from scripts.render_final_results import render
    render(OUT,tables,overview,protocol)
    save_json(ROOT/'records'/protocol['id']/'acceptance.json',dict(status='accepted',protocol_sha256=args.digest,
        models=90,score_rows=overview['score_rows'],pixel_audits=len(audits),report_sha256=file_hash(OUT/'report.md'),
        output_sha256={p.relative_to(ROOT).as_posix():file_hash(p) for p in sorted(OUT.iterdir())
                       if p.is_file() and p.suffix in {'.csv','.json','.png','.md'}}))
    print(overview)

if __name__=='__main__':main()
