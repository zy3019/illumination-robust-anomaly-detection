"""Freeze the complete accepted model set before opening any final test image."""
import json
import shutil
import sys
from pathlib import Path
from datetime import datetime,timezone,timedelta
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from anomaly.data import file_hash,save_json
from scripts.final_protocol import read,require,environment,load_locked
from scripts.summarize_dev_stability import collect


def main():
    # Existing full development verification reads only development outputs.
    collect()
    parity=read(ROOT/'records/formal_preflight/parity.json')
    require(parity['status']=='passed' and parity['final_test_evaluated'] is False,'Missing development parity')
    require(parity['environment']==environment(),'Preflight environment differs')
    for path,digest in parity['source_sha256'].items():require(file_hash(ROOT/path)==digest,'Preflight code changed: '+path)
    reference=read(ROOT/'records/dev_matrix_v1/plan.json')
    jobs=[]
    for matrix in ['dev_matrix_v1','dev_ab_extra_seeds_v1','dev_c_ablation_v1']:
        plan=read(ROOT/'records'/matrix/'plan.json');status=read(ROOT/'outputs'/matrix/'status.json')
        require(status['status']=='complete','Incomplete development matrix')
        for job in plan['jobs']:
            folder=ROOT/status['jobs'][job['id']]['output']
            frozen={ (folder/name).relative_to(ROOT).as_posix():file_hash(folder/name) for name in
                    ['model.pt','config.json','summary.json','threshold.json','sample_lists.json','training_views.json','calibration_scores.csv','provenance.json'] }
            jobs.append({**job,'source_output':folder.relative_to(ROOT).as_posix(),'frozen_files':frozen})
    jobs.sort(key=lambda j:(['bottle','metal_nut','tile'].index(j['config']['category']),
                           [5,10,25,'all'].index(j['config']['normal_images']),j['config']['seed'],j['config']['method']))
    base=jobs[0]['config'];manifest=read(ROOT/base['manifest'])
    final_samples={c:[r for r in manifest['rows'] if r['category']==c and r['split']=='final_test'] for c in ['bottle','metal_nut','tile']}
    sources=list(reference['core_sha256'])+[
        'scripts/final_protocol.py','scripts/final_metrics.py','scripts/evaluate_frozen_model.py',
        'scripts/run_final_matrix.py','scripts/summarize_final_results.py','scripts/render_final_results.py',
        'scripts/development_statistics.py','scripts/summarize_dev_stability.py','scripts/run_dev_matrix.py',
        'scripts/validate_final_evaluator.py','scripts/lock_final_protocol.py',
        'tests/test_final_evaluation.py','tests/test_development_statistics.py','tests/test_pipeline.py']
    for path,digest in reference['core_sha256'].items():require(file_hash(ROOT/path)==digest,'Development core changed')
    document='docs/formal_protocol_v1.md'
    assets=[base['manifest'],base['weights'],document,'records/formal_preflight/parity.json',
            'records/dev_stability_v1/acceptance.json','records/dev_followup_plan.json']
    protocol=dict(id='formal_v1',status='locked',locked_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),
        evaluation_split='final_test',environment=environment(),manifest=base['manifest'],
        conditions=reference['matrix']['conditions'],jobs=jobs,final_samples=final_samples,
        frozen_model_reuse=True,threshold_recalibration=False,
        source_sha256={p:file_hash(ROOT/p) for p in sources},assets_sha256={p:file_hash(ROOT/p) for p in assets},
        pixel_audit_jobs=[f'{c}_n5_A_seed17' for c in final_samples],pixel_audit_conditions=['clean','brightness_0.5'],
        description='All 90 accepted development models frozen without winner selection; custom final split; full conditions; fixed clean thresholds.',
        human_protocol=document,external_registration=False)
    folder=ROOT/'records/formal_v1'
    require(not folder.exists(),'Protocol already exists; never overwrite a lock')
    require(not (ROOT/'outputs/formal_v1').exists(),'Final outputs already exist before lock')
    folder.mkdir(parents=True)
    for relative in [*sources,*assets]:
        target=folder/'snapshot'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/relative,target)
    path=folder/'protocol.json'
    with path.open('x',encoding='utf-8') as stream:json.dump(protocol,stream,ensure_ascii=False,indent=2);stream.write('\n')
    digest=file_hash(path)
    with (folder/'protocol.sha256').open('x',encoding='ascii') as stream:stream.write(digest+'\n')
    load_locked(path,digest)
    save_json(folder/'lock_receipt.json',dict(status='locked_and_verified',protocol_sha256=digest,locked_at=protocol['locked_at'],
        jobs=90,final_images={c:len(r) for c,r in final_samples.items()},final_test_evaluated_at_lock=False))
    print('LOCKED SHA-256: '+digest,flush=True)

if __name__=='__main__':main()
