"""Readable compact scientific panels for the paper; uses accepted arrays only."""
import csv
import os
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'data/cache/matplotlib'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from anomaly.data import file_hash,load_image,load_mask,perturb
from scripts.final_protocol import read

def main():
    folder=ROOT/'reports/formal_v1/failure_analysis'
    provenance=read(folder/'provenance.json')
    with (folder/'selected_cases.csv').open(encoding='utf-8') as f:cases=list(csv.DictReader(f))
    assert file_hash(folder/'selected_cases.csv')==provenance['outputs_sha256']['selected_cases.csv']
    protocol=read(ROOT/'records/formal_v1/protocol.json');status=read(ROOT/'outputs/formal_v1/status.json')
    out=ROOT/'reports/research_report_v1/figures';out.mkdir(parents=True,exist_ok=True)
    for case_id in ['bottle_rescued_fp','metal_nut_persistent_fn','tile_introduced_fn']:
        case=next(r for r in cases if r['case_id']==case_id);cat,n,seed=case['category'],case['budget'],int(case['seed'])
        sample=next(r for r in protocol['final_samples'][cat] if r['path']==case['path'])
        condition=next(c for c in protocol['conditions'] if c['name']==case['condition'])
        j=next(j for j in protocol['jobs'] if j['id']==f'{cat}_n{n}_A_seed{seed}')
        cfg=j['config'];image=load_image(ROOT/cfg['data_root'],sample,224);mask=load_mask(ROOT/cfg['data_root'],sample,224)
        image=perturb(image,condition['operation'],condition['factor']).permute(1,2,0).numpy()
        maps=[]
        for method in ['A','B']:
            entry=status['jobs'][f'{cat}_n{n}_{method}_seed{seed}'];run=ROOT/entry['output']
            assert file_hash(run/'integrity.json')==entry['integrity_sha256']
            integrity=read(run/'integrity.json');rel=f"maps/{case['condition']}/{case['defect']}_{Path(case['path']).stem}.npy"
            assert file_hash(run/rel)==integrity[rel]
            assert file_hash(run/'threshold.json')==integrity['threshold.json']
            maps.append(np.load(run/rel)/read(run/'threshold.json')['threshold'])
        fig,axes=plt.subplots(1,3,figsize=(10.8,3.7),layout='constrained')
        axes[0].imshow(image)
        if mask.any():axes[0].contour(mask,levels=[.5],colors=['cyan'],linewidths=1.2)
        axes[0].set_title(case['condition']+' input',fontsize=12)
        for ax,amap,method in zip(axes[1:],maps,['A','B']):
            im=ax.imshow(amap,cmap='magma',vmin=0,vmax=max(1.2,*[m.max() for m in maps]))
            if mask.any():ax.contour(mask,levels=[.5],colors=['cyan'],linewidths=1.2)
            ax.set_title(f"{method}: {case[method+'_outcome']} | s/t={float(case[method+'_ratio']):.6f}",fontsize=12)
        for ax in axes:ax.set_xticks([]);ax.set_yticks([])
        cb=fig.colorbar(im,ax=axes[1:],shrink=.85);cb.ax.tick_params(labelsize=10)
        cb.set_label('Map / own image threshold',fontsize=10)
        fig.savefig(out/(case_id+'.png'),dpi=160);plt.close(fig)
    def rows(name):
        with (ROOT/'reports/formal_v1'/name).open(encoding='utf-8') as f:return list(csv.DictReader(f))
    def one(data,**keys):
        found=[r for r in data if all(r[k]==str(v) for k,v in keys.items())]
        assert len(found)==1
        return found[0]
    budgets=['5','10','25','all'];summary=rows('macro_severity_seed_summary.csv');seed_rows=rows('macro_severity_by_seed.csv')
    plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(10.2,4),layout='constrained')
    for ax,(severity,metric,title) in zip(axes,[('clean','image_auroc','Clean image AUROC'),('stress','image_auroc_drop','Stress AUROC drop')]):
        for method,color in [('A','#2563eb'),('B','#d97706')]:
            for seed in [17,42,2026]:
                ax.plot(range(4),[float(one(seed_rows,budget=n,method=method,seed=seed,severity=severity)[metric]) for n in budgets],color=color,alpha=.18)
            r=[one(summary,budget=n,method=method,severity=severity) for n in budgets]
            ax.errorbar(np.arange(4)+(-.025 if method=='A' else .025),[float(x[metric+'_mean']) for x in r],yerr=[float(x[metric+'_std']) for x in r],label=method,color=color,marker='o',capsize=3)
        ax.set_xticks(range(4),budgets);ax.set(title=title,xlabel='Original-image budget');ax.grid(alpha=.15);ax.legend()
    fig.savefig(out/'budget_summary.png',dpi=160);plt.close(fig)
    data=rows('macro_severity_paired_seed_summary.csv')
    fig,axes=plt.subplots(2,2,figsize=(9.7,6.1))
    for ax,(metric,label,scale) in zip(axes.ravel(),[('image_auroc','Image AUROC',1),('pixel_auroc','Pixel AUROC',1),('fpr','FPR (percentage points)',100),('recall','Recall (percentage points)',100)]):
        for i,severity in enumerate(['clean','mild','moderate','stress']):
            r=[one(data,budget=n,comparison='B-A',severity=severity) for n in budgets]
            ax.errorbar(np.arange(4)+(i-1.5)*.055,[float(x[metric+'_mean'])*scale for x in r],yerr=[float(x[metric+'_std'])*scale for x in r],label=severity,marker='o',markersize=3,capsize=2)
        ax.axhline(0,color='black',ls=':',lw=.8);ax.grid(alpha=.15);ax.set_xticks(range(4),budgets);ax.set_title(label,fontsize=12);ax.set_ylabel('B - A');ax.set_xlabel('Original-image budget')
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,ncol=4,loc='upper center',frameon=False)
    fig.tight_layout(rect=(0,0,1,.92));fig.savefig(out/'paired_effects.png',dpi=160);plt.close(fig)
    print('Three compact case panels and two legible summary figures rendered.')

if __name__=='__main__':main()
