"""Run all frozen final jobs sequentially with immutable outputs and resumable checks."""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime,timezone,timedelta
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from anomaly.data import file_hash,save_json
from scripts.final_protocol import read,require,load_locked,verify_files


def now():return datetime.now(timezone(timedelta(hours=8))).isoformat()

def atomic(path,value):
    temp=path.with_suffix('.tmp');save_json(temp,value);temp.replace(path)

def verify_result(folder,job,protocol,digest,expected_integrity=None):
    if expected_integrity:
        require(file_hash(folder/'integrity.json')==expected_integrity,'Output integrity manifest changed')
    integrity=read(folder/'integrity.json')
    actual={p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file() and p.name!='integrity.json'}
    require(actual==set(integrity),'Output file set differs')
    verify_files(integrity,folder)
    summary=read(folder/'summary.json')
    require(summary['status']=='final_complete' and summary['evaluation_split']=='final_test','Not a final result')
    require(summary['protocol_sha256']==digest and summary['job_id']==job['id'],'Output protocol/job mismatch')
    require(summary['memory_vectors']==1000 and read(folder/'config.json')==job['config'],'Output model mismatch')
    samples=protocol['final_samples'][job['config']['category']]
    require(read(folder/'sample_lists.json')['final_test']==samples,'Final samples mismatch')
    require(summary['evaluation_images']==len(samples),'Final image count mismatch')
    threshold=read(ROOT/job['source_output']/'threshold.json')
    require(summary['threshold']==threshold==read(folder/'threshold.json'),'Frozen threshold changed')
    require({m['condition'] for m in summary['metrics']}=={c['name'] for c in job['config']['conditions']} and len(summary['metrics'])==13,'Condition mismatch')
    import csv
    with (folder/'scores.csv').open(encoding='utf-8') as stream:scores=list(csv.DictReader(stream))
    expected={(c['name'],r['path']) for c in job['config']['conditions'] for r in samples}
    require(len(scores)==len(expected) and {(r['condition'],r['path']) for r in scores}==expected,'Missing or duplicate scores')
    require(all(float(r['threshold'])==threshold['threshold'] for r in scores),'Nonconstant threshold')
    for c in job['config']['conditions']:
        for row in samples:
            require((folder/'maps'/c['name']/f"{row['defect']}_{Path(row['path']).stem}.npy").is_file(),'Missing map')
    return summary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--protocol',default='records/formal_v1/protocol.json')
    parser.add_argument('--digest',required=True)
    args=parser.parse_args()
    path=ROOT/args.protocol
    protocol,_=load_locked(path,args.digest)
    folder=ROOT/'outputs'/protocol['id'];folder.mkdir(parents=True,exist_ok=True)
    require(shutil.disk_usage(ROOT).free>30*2**30,'Need 30 GiB free for final output/resume')
    lock=folder/'RUNNING.lock'
    with lock.open('x') as stream:stream.write(str(os.getpid()))
    try:
        status_path=folder/'status.json'
        status=read(status_path) if status_path.exists() else dict(protocol_sha256=args.digest,started_at=now(),jobs={})
        require(status['protocol_sha256']==args.digest,'Resume protocol differs')
        for index,job in enumerate(protocol['jobs'],1):
            load_locked(path,args.digest,job['id'])
            previous=status['jobs'].get(job['id'],{})
            if previous.get('status')=='complete':
                verify_result(ROOT/previous['output'],job,protocol,args.digest,previous['integrity_sha256'])
                print(f'[{index}/90] Verified completed {job["id"]}',flush=True);continue
            attempt=1
            while (folder/f'{job["id"]}_attempt{attempt:02}').exists():attempt+=1
            target=folder/f'{job["id"]}_attempt{attempt:02}'
            log=folder/f'{job["id"]}_attempt{attempt:02}.log'
            entry=dict(status='running',attempt=attempt,output=target.relative_to(ROOT).as_posix(),log=log.relative_to(ROOT).as_posix(),started_at=now())
            status['jobs'][job['id']]=entry;atomic(status_path,status)
            print(f'[{index}/90] Running {job["id"]}',flush=True)
            start=time.perf_counter()
            with log.open('w',encoding='utf-8') as stream:
                result=subprocess.run([sys.executable,'-B','-u','scripts/evaluate_frozen_model.py',
                    '--protocol',str(path),'--digest',args.digest,'--job',job['id'],'--output',str(target)],
                    cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            entry['process_wall_seconds']=time.perf_counter()-start
            if result.returncode:
                entry['status']='failed';atomic(status_path,status)
                raise RuntimeError('Final job failed; retain attempt and inspect '+str(log))
            verify_result(target,job,protocol,args.digest)
            entry.update(status='complete',finished_at=now(),integrity_sha256=file_hash(target/'integrity.json'),
                         artifact_bytes=sum(p.stat().st_size for p in target.rglob('*') if p.is_file()))
            atomic(status_path,status)
            print(f'[{index}/90] Complete {job["id"]}: {entry["process_wall_seconds"]:.1f}s',flush=True)
        status.update(status='complete',finished_at=now());atomic(status_path,status)
        save_json(ROOT/'records'/protocol['id']/'execution.json',status)
        print('All 90 final jobs completed',flush=True)
    finally:lock.unlink(missing_ok=True)

if __name__=='__main__':main()
