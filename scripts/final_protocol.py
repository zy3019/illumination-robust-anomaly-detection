"""Protocol integrity checks performed before any held-out image access."""
import json
from pathlib import Path
import platform
import numpy as np
import torch
import torchvision
from PIL import __version__ as pillow_version
from anomaly.data import file_hash, contained, validate_manifest

ROOT=Path(__file__).resolve().parents[1]

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def require(ok, message):
    if not ok: raise ValueError(message)

def environment():
    return dict(python=platform.python_version(),torch=torch.__version__,torchvision=torchvision.__version__,numpy=np.__version__,pillow=pillow_version)

def verify_files(files,root=ROOT):
    for relative,expected in files.items():
        path=contained(root,relative)
        require(file_hash(path)==expected,'Locked file changed: '+relative)

def load_locked(path,expected_digest,job_id=None,root=ROOT):
    path=Path(path)
    require(bool(expected_digest) and file_hash(path)==expected_digest,'Protocol digest mismatch')
    protocol=read(path)
    require(protocol['status']=='locked' and protocol['evaluation_split']=='final_test','Final protocol is not locked')
    require(protocol['environment']==environment(),'Locked inference environment changed')
    verify_files(protocol['source_sha256'],root)
    verify_files(protocol['assets_sha256'],root)
    manifest=read(contained(root,protocol['manifest']))
    validate_manifest(manifest)
    expected={(c,str(n),m,s) for c in ['bottle','metal_nut','tile'] for n in [5,10,25,'all'] for s in [17,42,2026]
              for m in (['A','B','C'] if n in [5,25] else ['A','B'])}
    actual=[(j['config']['category'],str(j['config']['normal_images']),j['config']['method'],j['config']['seed']) for j in protocol['jobs']]
    require(len(actual)==90 and set(actual)==expected,'Final matrix is incomplete or contains duplicates')
    require(len({j['id'] for j in protocol['jobs']})==90,'Duplicate job IDs')
    selected=[j for j in protocol['jobs'] if j['id']==job_id] if job_id else protocol['jobs']
    require(bool(selected),'Unknown final job')
    for job in selected:
        require(job['config']['evaluation_split']=='dev','Frozen model config must retain training provenance')
        verify_files(job['frozen_files'],root)
        rows=[r for r in manifest['rows'] if r['category']==job['config']['category'] and r['split']=='final_test']
        require(rows==protocol['final_samples'][job['config']['category']],'Locked final sample list differs')
        require({r['label'] for r in rows}=={0,1},'Final split needs both classes')
    return protocol, (selected[0] if job_id else None)
