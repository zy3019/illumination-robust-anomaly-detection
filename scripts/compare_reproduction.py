"""Compare a new development run with frozen evidence, excluding clocks and metadata."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from anomaly.data import file_hash, save_json


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def compare(actual, reference):
    actual,reference=(Path(os.path.abspath(p)) for p in (actual,reference))
    for name in ['config.json','sample_lists.json','training_views.json','threshold.json']:
        if read(actual/name) != read(reference/name):
            raise ValueError('Reproduction differs: '+name)
    a=torch.load(actual/'model.pt',map_location='cpu',weights_only=True)
    b=torch.load(reference/'model.pt',map_location='cpu',weights_only=True)
    for key in ['memory_bank','selected_indices']:
        if not torch.equal(a[key],b[key]):
            raise ValueError('Frozen tensor differs: '+key)
    def rows(folder,name):
        with (folder/name).open(encoding='utf-8') as stream:
            return list(csv.DictReader(stream))
    new,old=rows(actual,'scores.csv'),rows(reference,'scores.csv')
    if len(new)!=len(old):raise ValueError('Score count differs')
    max_delta=0.
    for x,y in zip(new,old):
        if {k:v for k,v in x.items() if k!='seconds'} != {k:v for k,v in y.items() if k!='seconds'}:
            raise ValueError('Per-image score/decision differs: '+x['path'])
        max_delta=max(max_delta,abs(float(x['score'])-float(y['score'])))
    am,bm=read(actual/'summary.json')['metrics'],read(reference/'summary.json')['metrics']
    if len(am)!=len(bm):raise ValueError('Metric count differs')
    for x,y in zip(am,bm):
        for key in ['condition','image_auroc','pixel_auroc','fpr','recall','false_positives','true_positives']:
            if x[key]!=y[key]:raise ValueError('Metric differs: '+key)
    return dict(status='passed',evaluation_split='dev',score_rows=len(new),conditions=len(am),
                memory_bank_exact=True,selected_indices_exact=True,threshold_exact=True,
                score_max_absolute_difference=max_delta,metrics_exact=True,
                actual=str(actual.resolve()),reference=str(reference.resolve()),
                actual_model_sha256=file_hash(actual/'model.pt'),reference_model_sha256=file_hash(reference/'model.pt'),
                comparison='Tensor/data values, not serialization bytes or wall clocks',
                personal_independence='Not assessed by automated comparison')


def main():
    p=argparse.ArgumentParser();p.add_argument('--actual',required=True,type=Path)
    p.add_argument('--reference',required=True,type=Path);p.add_argument('--receipt',required=True,type=Path)
    args=p.parse_args()
    result=compare(args.actual,args.reference)
    if args.receipt.exists():raise FileExistsError('Choose a new receipt path')
    save_json(args.receipt,result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
