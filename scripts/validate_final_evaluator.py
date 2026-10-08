"""Validate the final evaluator against development outputs without final image access."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from anomaly.data import file_hash,save_json
from scripts.final_protocol import read,require,environment
from scripts.evaluate_frozen_model import evaluate
from scripts.summarize_dev_stability import read_csv

out=ROOT/'outputs/formal_preflight'
out.mkdir(parents=True,exist_ok=True)
selected=[('dev_matrix_v1','bottle_n5_B_seed42'),('dev_c_ablation_v1','metal_nut_n25_C_seed17'),
          ('dev_ab_extra_seeds_v1','tile_nall_A_seed2026')]
results=[]
for matrix,job_id in selected:
    plan=read(ROOT/'records'/matrix/'plan.json');status=read(ROOT/'outputs'/matrix/'status.json')
    job=next(j for j in plan['jobs'] if j['id']==job_id)
    job={**job,'source_output':status['jobs'][job_id]['output']}
    target=out/job_id
    if target.exists():
        raise ValueError('Preflight output exists; preserve it and investigate rather than overwrite')
    actual=evaluate(job,{},target,split='dev')
    old=ROOT/job['source_output'];reference=read(old/'summary.json')
    for a,b in zip(actual['metrics'],reference['metrics']):
        for key in ['condition','image_auroc','false_positives','true_positives','normal_images','anomalous_images','fpr','recall','image_auroc_drop','fpr_increase']:
            require(a[key]==b[key],f'{job_id}: mismatch {key}')
        require(abs(a['pixel_auroc']-b['pixel_auroc'])<1e-12,'Pixel AUROC mismatch')
    old_scores=read_csv(old/'scores.csv'); new_scores=read_csv(target/'scores.csv')
    require(len(old_scores)==len(new_scores),'Score count mismatch')
    for a,b in zip(old_scores,new_scores):
        require(all(a[k]==b[k] for k in ['condition','path','label','score','threshold','predicted_anomaly']),'Saved score mismatch')
    maps=list((target/'maps').rglob('*.npy'))
    for path in maps:
        require(np.array_equal(np.load(path),np.load(old/path.relative_to(target))),'Map mismatch')
    results.append(dict(job=job_id,score_rows=len(new_scores),exact_maps=len(maps),pixel_auroc_max_diff=max(
        abs(a['pixel_auroc']-b['pixel_auroc']) for a,b in zip(actual['metrics'],reference['metrics']))))
    print('PARITY PASSED:',job_id,flush=True)
save_json(ROOT/'records/formal_preflight/parity.json',dict(status='passed',final_test_evaluated=False,results=results,
    environment=environment(),source_sha256={p:file_hash(ROOT/p) for p in ['scripts/evaluate_frozen_model.py','scripts/final_metrics.py','scripts/final_protocol.py']}))
