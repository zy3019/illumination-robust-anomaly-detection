"""Evaluate a frozen memory bank; final access requires a pinned protocol digest."""
import argparse
import json
import os
import sys
import time
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from anomaly.data import load_image,load_mask,perturb,save_json,file_hash
from anomaly.model import PatchFeatures,predict
from anomaly.metrics import classification_metrics
from anomaly.run import sync,phase_start,phase_end,process_peak_memory,write_csv,visual
from scripts.final_protocol import read,require,load_locked
from scripts.final_metrics import exact_pixel_auroc


def evaluate(job,protocol,output,split='final_test',digest=None):
    # Public function also enforces the guard: callers cannot bypass it through imports.
    if split=='final_test':
        require(protocol.get('_path') is not None,'Missing locked protocol path')
        checked, frozen=load_locked(protocol['_path'],digest,job['id'])
        require(frozen==job,'Job differs from locked configuration')
        protocol=checked
    elif split!='dev':
        raise ValueError('Unknown evaluation split')
    cfg=job['config']
    source=ROOT/job['source_output']
    model=torch.load(source/'model.pt',map_location='cpu',weights_only=True)
    require(model['config']==cfg,'Saved model config mismatch')
    summary=read(source/'summary.json')
    threshold=summary['threshold']['threshold']
    require(model['threshold']==threshold,'Saved model threshold mismatch')
    require(tuple(model['memory_bank'].shape)==(1000,384),'Wrong memory capacity')
    manifest=read(ROOT/cfg['manifest'])
    require(file_hash(ROOT/cfg['manifest'])==model['manifest_sha256'],'Model manifest mismatch')
    require(file_hash(ROOT/cfg['weights'])==model['weights_sha256'],'Model weights mismatch')
    rows=[r for r in manifest['rows'] if r['category']==cfg['category'] and r['split']==split]
    require({r['label'] for r in rows}=={0,1},'Incomplete evaluation labels')
    out=Path(output)
    out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    torch.set_num_threads(4)
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True)
    device=cfg['device']
    require(device=='cuda' and torch.cuda.is_available(),'Locked CUDA device unavailable')
    extractor=PatchFeatures(ROOT/cfg['weights'],device)
    bank=model['memory_bank'].to(device)
    samples=read(source/'sample_lists.json')
    save_json(out/'config.json',cfg)
    save_json(out/'sample_lists.json',{'train':samples['train'],'calibration':samples['calibration'],split:rows})
    save_json(out/'threshold.json',summary['threshold'])
    from datetime import datetime,timezone,timedelta
    provenance=dict(protocol_sha256=digest,job_id=job['id'],evaluation_split=split,
                    source_model_sha256=file_hash(source/'model.pt'),source_output=job['source_output'],
                    started_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),
                    bank_reused=True,threshold_reused=True,gpu=torch.cuda.get_device_name(),
                    source_sha256={p.relative_to(ROOT).as_posix():file_hash(p) for p in
                        [Path(__file__),ROOT/'scripts/final_metrics.py',ROOT/'anomaly/model.py']})
    save_json(out/'provenance.json',provenance)
    # This marker is written BEFORE the first final image is opened, including failed attempts.
    save_json(out/'exposure_started.json',dict(split=split,protocol_sha256=digest,at=provenance['started_at']))
    start=phase_start(device)
    images=[load_image(ROOT/cfg['data_root'],r,cfg['image_size']) for r in rows]
    masks=[load_mask(ROOT/cfg['data_root'],r,cfg['image_size']) for r in rows]
    resources={'input_loading':phase_end(start,device)}
    labels=[r['label'] for r in rows]
    pixel_labels=np.concatenate([m.ravel() for m in masks])
    for _ in range(3): predict(extractor,images[0].to(device),bank)
    sync(device)
    benchmarks=[]
    for row,image in zip(rows[:3],images[:3]):
        durations=[]
        for _ in range(10):
            sync(device); tick=time.perf_counter()
            predict(extractor,image.to(device),bank)
            sync(device); durations.append(time.perf_counter()-tick)
        benchmarks.append(dict(path=row['path'],seconds=durations,median_seconds=float(np.median(durations))))
    save_json(out/'inference_benchmark.json',dict(warmups=3,repeats=10,images=benchmarks,
        scope='CPU tensor to GPU, features, 1NN, map and CPU output; excludes disk/PIL'))
    all_scores,metrics=[],[]
    start=phase_start(device)
    for condition in cfg['conditions']:
        folder=out/'maps'/condition['name']; folder.mkdir(parents=True)
        scores,maps,timings=[],[],[]
        shown=set()
        for row,original,mask in zip(rows,images,masks):
            image=perturb(original,condition['operation'],condition['factor'])
            sync(device); tick=time.perf_counter()
            score,anomaly_map=predict(extractor,image.to(device),bank)
            sync(device); elapsed=time.perf_counter()-tick
            identifier=f"{row['defect']}_{Path(row['path']).stem}"
            np.save(folder/(identifier+'.npy'),anomaly_map)
            if row['defect'] not in shown:
                visual(image,anomaly_map,threshold,mask,folder/(identifier+'.png'))
                shown.add(row['defect'])
            scores.append(score); maps.append(anomaly_map.ravel()); timings.append(elapsed)
            all_scores.append(dict(condition=condition['name'],path=row['path'],label=row['label'],defect=row['defect'],
                                   score=score,threshold=threshold,predicted_anomaly=int(score>threshold),seconds=elapsed))
        result=dict(condition=condition['name'],**classification_metrics(labels,scores,threshold),
                    pixel_auroc=exact_pixel_auroc(pixel_labels,np.concatenate(maps)),
                    inference_seconds_median=float(np.median(timings)),timed_images=len(rows))
        metrics.append(result)
        print(json.dumps(result),flush=True)
    resources['evaluation']=phase_end(start,device)
    resources['evaluation']['process_peak_working_set_bytes_so_far']=process_peak_memory()
    clean=next(m for m in metrics if m['condition']=='clean')
    for m in metrics:
        m['image_auroc_drop']=clean['image_auroc']-m['image_auroc']
        m['fpr_increase']=m['fpr']-clean['fpr']
    write_csv(out/'scores.csv',all_scores); write_csv(out/'metrics.csv',metrics)
    result=dict(status='final_complete' if split=='final_test' else 'development_parity',evaluation_split=split,
                final_test_evaluated=split=='final_test',protocol_sha256=digest,job_id=job['id'],
                category=cfg['category'],method=cfg['method'],seed=cfg['seed'],training_originals=summary['training_originals'],
                evaluation_images=len(rows),memory_vectors=len(bank),threshold=summary['threshold'],metrics=metrics,
                resources=resources,run_wall_seconds=time.perf_counter()-started,
                repeated_inference_median_seconds=float(np.median([t for b in benchmarks for t in b['seconds']])))
    save_json(out/'summary.json',result)
    # Hash every output including all maps. Enables resume to detect accidental changes.
    integrity={p.relative_to(out).as_posix():file_hash(p) for p in sorted(out.rglob('*')) if p.is_file()}
    save_json(out/'integrity.json',integrity)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--protocol',required=True);p.add_argument('--digest',required=True)
    p.add_argument('--job',required=True);p.add_argument('--output',required=True)
    args=p.parse_args()
    protocol,job=load_locked(args.protocol,args.digest,args.job)
    protocol['_path']=args.protocol
    evaluate(job,protocol,args.output,digest=args.digest)

if __name__=='__main__':main()
