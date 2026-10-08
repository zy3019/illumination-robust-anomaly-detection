"""Post-hoc diagnosis of accepted final scores/maps; never fits or runs a model."""
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from datetime import datetime, timezone, timedelta

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'data/cache/matplotlib'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from anomaly.data import file_hash, load_image, load_mask, perturb, save_json
from scripts.final_protocol import load_locked, read, require

DIGEST = '58efbb9a2e54a4e276d062235f141f687a0aeaa84f26e48eb34cfe34b82a6885'
OUT = ROOT / 'reports/formal_v1/failure_analysis'
CATS = ['bottle', 'metal_nut', 'tile']
BUDGETS = ['5', '10', '25', 'all']
ROLES = ['persistent_fp', 'rescued_fp', 'introduced_fp', 'persistent_fn', 'rescued_fn', 'introduced_fn']
ROLE_CN = dict(zip(ROLES, ['A/B均误报', 'B纠正A误报', 'B新增误报', 'A/B均漏检', 'B纠正A漏检', 'B新增漏检']))


def csv_write(path, rows):
    require(bool(rows), 'Empty table: ' + str(path))
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def outcome(label, predicted):
    return ('TP' if predicted else 'FN') if label else ('FP' if predicted else 'TN')


def pair_role(label, a, b):
    if a == label and b == label:
        return 'both_correct'
    kind = 'fn' if label else 'fp'
    return ('persistent_' if a != label and b != label else 'rescued_' if a != label else 'introduced_') + kind


def main():
    protocol, _ = load_locked(ROOT / 'records/formal_v1/protocol.json', DIGEST)
    status = read(ROOT / 'outputs/formal_v1/status.json')
    acceptance = read(ROOT / 'records/formal_v1/acceptance.json')
    require(status['status'] == 'complete' and acceptance['status'] == 'accepted', 'Final matrix not accepted')
    require(acceptance['protocol_sha256'] == DIGEST, 'Acceptance protocol mismatch')
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = {}

    def track(path, expected=None):
        sha = file_hash(path)
        if expected is not None:
            require(sha == expected, 'Changed input: ' + str(path))
        inputs[path.relative_to(ROOT).as_posix()] = sha

    for name in ['records/formal_v1/protocol.json', 'records/formal_v1/acceptance.json',
                 'outputs/formal_v1/status.json', 'reports/formal_v1/report.md', 'scripts/analyze_final_failures.py']:
        track(ROOT / name)
    require(inputs['reports/formal_v1/report.md'] == acceptance['report_sha256'], 'Formal report changed')
    records, units, strata = [], {}, []
    for job in protocol['jobs']:
        cfg = job['config']; entry = status['jobs'][job['id']]; folder = ROOT / entry['output']
        require(entry['status'] == 'complete', 'Incomplete job')
        track(folder / 'integrity.json', entry['integrity_sha256'])
        integrity = read(folder / 'integrity.json')
        for name in ['scores.csv', 'summary.json', 'threshold.json', 'config.json', 'sample_lists.json']:
            track(folder / name, integrity[name])
        require(read(folder / 'config.json') == cfg, 'Configuration mismatch')
        summary = read(folder / 'summary.json')
        require(summary['protocol_sha256'] == DIGEST, 'Output protocol mismatch')
        samples = {r['path']: r for r in protocol['final_samples'][cfg['category']]}
        threshold = read(folder / 'threshold.json')['threshold']
        require(threshold > 0, 'Ratio display requires positive threshold')
        with (folder / 'scores.csv').open(encoding='utf-8') as f:
            scores = list(csv.DictReader(f))
        score_map = {(s['condition'], s['path']): s for s in scores}
        expected = {(c['name'], path) for c in cfg['conditions'] for path in samples}
        require(set(score_map) == expected and len(scores) == len(expected), 'Score coverage mismatch')
        meta = dict(category=cfg['category'], budget=str(cfg['normal_images']), method=cfg['method'], seed=cfg['seed'])
        unit = dict(folder=folder, integrity=integrity, cfg=cfg, samples=samples, rows={})
        units[(meta['category'], meta['budget'], meta['method'], meta['seed'])] = unit
        for s in scores:
            sample = samples[s['path']]; label = int(s['label']); score = float(s['score'])
            pred = int(s['predicted_anomaly']); clean = score_map['clean', s['path']]
            require(label == sample['label'] and s['defect'] == sample['defect'], 'Label mismatch')
            require(np.isfinite(score) and float(s['threshold']) == threshold and pred == int(score > threshold), 'Decision mismatch')
            error = pred != label; clean_pred = int(clean['predicted_anomaly']); clean_error = clean_pred != label
            transition = ('persistent_error' if error else 'recovered') if clean_error else ('new_error' if error else 'correct_both')
            row = dict(**meta, job=job['id'], condition=s['condition'], path=s['path'], defect=s['defect'], label=label,
                       score=score, threshold=threshold, score_over_threshold=score/threshold,
                       predicted=pred, outcome=outcome(label,pred), clean_score=float(clean['score']),
                       clean_predicted=clean_pred, clean_outcome=outcome(label,clean_pred), transition_from_clean=transition)
            records.append(row); unit['rows'][s['condition'],s['path']] = row
        for m in summary['metrics']:
            group = [r for r in unit['rows'].values() if r['condition'] == m['condition']]
            require(sum(r['outcome']=='FP' for r in group)==m['false_positives'], 'FP audit')
            require(sum(r['outcome']=='TP' for r in group)==m['true_positives'], 'TP audit')
            for defect in sorted({r['defect'] for r in group}):
                sub = [r for r in group if r['defect']==defect]
                count = sum(r['outcome'] in ['FP','FN'] for r in sub)
                strata.append(dict(**meta, condition=m['condition'], defect=defect, label=sub[0]['label'],
                                   images=len(sub), errors=count, error_rate=count/len(sub)))
    require(len(records)==98280, 'Unexpected total score count')
    pairs = []
    for cat in CATS:
        for n in BUDGETS:
            for seed in [17,42,2026]:
                a,b = (units[cat,n,m,seed] for m in ['A','B'])
                require(set(a['rows'])==set(b['rows']), 'Unpaired samples')
                for key, ar in a['rows'].items():
                    br=b['rows'][key]
                    pairs.append(dict(category=cat,budget=n,seed=seed,condition=key[0],path=key[1],
                                      label=ar['label'],defect=ar['defect'],role=pair_role(ar['label'],ar['predicted'],br['predicted']),
                                      A_outcome=ar['outcome'],B_outcome=br['outcome'],
                                      A_ratio=ar['score_over_threshold'],B_ratio=br['score_over_threshold']))
    require(len(pairs)==39312, 'Unexpected A/B paired comparison count')
    grouped=defaultdict(list)
    for p in pairs: grouped[p['category'],p['budget'],p['seed'],p['condition'],p['label']].append(p)
    transitions=[]
    for (cat,n,seed,condition,label), group in grouped.items():
        count=Counter(r['role'].split('_')[0] for r in group)
        a_err=sum(r['A_outcome'] in ['FP','FN'] for r in group)
        b_err=sum(r['B_outcome'] in ['FP','FN'] for r in group)
        require(b_err-a_err==count['introduced']-count['rescued'], 'Paired error identity failed')
        transitions.append(dict(category=cat,budget=n,seed=seed,condition=condition,label=label,images=len(group),
            A_errors=a_err,B_errors=b_err,both_wrong=count['persistent'],B_rescued=count['rescued'],
            B_introduced=count['introduced'],both_correct=count['both']))
    selected=[]; missing=[]
    condition_order={c['name']:i for i,c in enumerate(protocol['conditions'])}
    recurrence=Counter((p['category'],p['budget'],p['condition'],p['path'],p['role']) for p in pairs)
    for cat in CATS:
        for role in ROLES:
            pool=[p for p in pairs if p['category']==cat and p['role']==role]
            if not pool:
                missing.append(dict(category=cat,role=role));continue
            def rank(p):
                budgets=['all','25','10','5'] if p['label']==0 else ['5','10','25','all']
                return (-recurrence[cat,p['budget'],p['condition'],p['path'],role],
                        0 if p['condition'].endswith(('_0.5','_1.5')) else 1,
                        budgets.index(p['budget']),condition_order[p['condition']],p['path'],p['seed'])
            p=dict(min(pool,key=rank));p['case_id']=f'{cat}_{role}'
            p['same_pattern_seeds']=recurrence[cat,p['budget'],p['condition'],p['path'],role]
            p['candidate_comparisons']=len(pool)
            p['candidate_distinct_images']=len({r['path'] for r in pool})
            selected.append(p)
    for p in selected:
        cat,n,seed,condition=p['category'],p['budget'],p['seed'],p['condition']
        a,b=(units[cat,n,m,seed] for m in ['A','B']);sample=a['samples'][p['path']]
        cfg=a['cfg'];data_root=ROOT/cfg['data_root'];original=load_image(data_root,sample,cfg['image_size'])
        mask=load_mask(data_root,sample,cfg['image_size'])
        track(data_root/sample['path'],sample['sha256'])
        if sample['mask']:track(data_root/sample['mask'],sample['mask_sha256'])
        c=next(c for c in protocol['conditions'] if c['name']==condition)
        changed=perturb(original,c['operation'],c['factor'])
        raw=original if c['operation']=='clean' else original*c['factor'] if c['operation']=='brightness' else (original-original.mean((-2,-1),keepdim=True))*c['factor']+original.mean((-2,-1),keepdim=True)
        p['clipped_channel_fraction']=float(((raw<0)|(raw>1)).float().mean())
        p['mask_area_fraction']=float(mask.mean())
        maps={}
        for name in ['clean',condition]:
            for method,u in [('A',a),('B',b)]:
                rel=f"maps/{name}/{sample['defect']}_{Path(sample['path']).stem}.npy"
                track(u['folder']/rel,u['integrity'][rel])
                amap=np.load(u['folder']/rel)
                require(amap.shape==mask.shape and np.isfinite(amap).all(), 'Invalid selected map')
                maps[name,method]=amap/u['rows'][name,p['path']]['threshold']
        vmax=max(1.2,max(float(v.max()) for v in maps.values()))
        fig,axes=plt.subplots(2,4,figsize=(13.6,7.2),layout='constrained')
        for row,(name,tensor) in enumerate([('clean',original),(condition,changed)]):
            axes[row,0].imshow(tensor.permute(1,2,0).numpy());axes[row,0].set_title(name)
            axes[row,1].imshow(mask,cmap='gray',vmin=0,vmax=1);axes[row,1].set_title('Ground-truth defect mask')
            for col,(method,u) in enumerate([('A',a),('B',b)],2):
                ax=axes[row,col];im=ax.imshow(maps[name,method],cmap='magma',vmin=0,vmax=vmax)
                if mask.any():ax.contour(mask,levels=[.5],colors=['cyan'],linewidths=.7)
                r=u['rows'][name,p['path']]
                ax.set_title(f"{method}: {r['outcome']} | score/t={r['score_over_threshold']:.6f}")
            for ax in axes[row]:ax.set_xticks([]);ax.set_yticks([])
        fig.colorbar(im,ax=axes[:,2:].ravel().tolist(),shrink=.85,label='Map / own image threshold (not probability)')
        fig.suptitle(f"{p['case_id']} | n={n}, seed={seed}, recurrence={p['same_pattern_seeds']}/3\n{p['path']} | cyan: ground truth; common display scale within panel",fontsize=11)
        fig.savefig(OUT/(p['case_id']+'.png'),dpi=125);plt.close(fig)
        print('Rendered '+p['case_id'],flush=True)
    errors=[r for r in records if r['outcome'] in ['FP','FN']]
    for name,rows in [('all_scores',records),('errors',errors),('defect_strata',strata),('paired_cases',pairs),('paired_transitions',transitions),('selected_cases',selected)]:
        csv_write(OUT/(name+'.csv'),rows)
    # Every error occurrence is retained; grouped unique image counts are descriptive only.
    counts=[]
    for cat in CATS:
        for method in ['A','B','C']:
            group=[r for r in records if r['category']==cat and r['method']==method]
            fp=[r for r in group if r['outcome']=='FP'];fn=[r for r in group if r['outcome']=='FN']
            counts.append(dict(category=cat,method=method,scoring_events=len(group),fp_events=len(fp),fn_events=len(fn),
                               distinct_fp_images=len({r['path'] for r in fp}),distinct_fn_images=len({r['path'] for r in fn})))
    csv_write(OUT/'error_coverage.csv',counts)
    rules=dict(analysis_type='post-hoc descriptive diagnosis, not a new confirmatory test',
        coverage='all 90 accepted models; all 13 conditions; all 252 independent final images',
        roles=ROLE_CN,selection='One A/B example per category and role; first maximize recurrence across three seeds; then prefer stress conditions; FP budget order all,25,10,5; FN budget order 5,10,25,all; then frozen condition order, image path, seed ascending. No substitute if empty.',
        heatmaps='Saved bilinear maps divided by each model own image threshold; common color scale within each panel; not probabilities or pixel segmentation decisions. Image scores use raw patch maxima and may differ from interpolated map maxima.',
        missing_roles=missing)
    save_json(OUT/'selection_rules.json',rules)
    overview=dict(status='complete',analysis_type=rules['analysis_type'],models=90,scoring_events=len(records),
                  independent_images=252,error_events=len(errors),fp_events=sum(r['outcome']=='FP' for r in errors),
                  fn_events=sum(r['outcome']=='FN' for r in errors),paired_comparisons=len(pairs),selected_cases=len(selected),missing_roles=missing)
    save_json(OUT/'overview.json',overview)
    lines=['# 最终测试后的失败案例分析','',
        '这是 formal_v1 验收之后的描述性分析。读取既有分数和异常图，不重新推理、不修改模型或阈值；不能将案例解释当成新的确认性实验。','',
        f"已核对全部 90 个模型、{len(records):,} 条评分和 {len(pairs):,} 条 A/B 配对比较；误报事件 {overview['fp_events']:,} 次，漏检事件 {overview['fn_events']:,} 次。它们来自 252 张独立原图的重复评估，不能作为独立错误样本量或部署错误率。",'',
        '## 完整覆盖与选择规则','',
        '下表的不同图像数是各类别/方法内部的并集。A/B 有四种预算，C 只有两种预算，因此不能直接用累计事件数给方法排名。完整逐条件分母见 defect_strata.csv；配对纠正和新增错误见 paired_transitions.csv。','',
        '| 类别 | 方法 | 评分次数 | 误报次数 | 漏检次数 | 曾误报的不同正常图 | 曾漏检的不同异常图 |','|---|---|---:|---:|---:|---:|---:|']
    for r in counts:lines.append('| '+' | '.join(str(r[k]) for k in r)+' |')
    lines += ['', '每类别分别选 A/B 均误报、B 纠正误报、B 新增误报、A/B 均漏检、B 纠正漏检、B 新增漏检。首先优先三个种子中重复出现的相同模式，再优先压力条件；误报优先全量预算，漏检优先小预算；最后按冻结条件顺序、文件名和种子排序。该规则在最终结果已可见后制定，不是预注册规则。空类别不补选其他角色。','',
        '热力图使用保存的异常图除以各自图像阈值，同一面板共用色标。颜色不是概率，图像阈值不是像素分割阈值；插值后的异常图最大值也不必等于原始 patch 最大图像分数。青色轮廓是缺陷标注。','',
        '## 案例索引','', '| 案例 | 类型 | 预算 / 种子 | 条件 | 同模式种子 | A 比值 | B 比值 |','|---|---|---|---|---:|---:|---:|']
    for p in selected:lines.append(f"| [{p['case_id']}]({p['case_id']}.png) | {ROLE_CN[p['role']]} | {p['budget']} / {p['seed']} | {p['condition']} | {p['same_pattern_seeds']}/3 | {p['A_ratio']:.6f} | {p['B_ratio']:.6f} |")
    lines += ['', '每个案例的原图路径、候选数、缺陷面积、裁剪比例保存在 selected_cases.csv。所有候选保存在 paired_cases.csv，所有错误保存在 errors.csv；没有只保留图示案例。','',
        '## 可复现与边界','',
        '运行 `.\\.venv-audit\\Scripts\\python.exe -B scripts/analyze_final_failures.py`。入口先检查锁定协议与验收状态；对消费的评分、元数据及选中异常图校验原 integrity.json 哈希，对输入图像/掩码校验原清单。未重复重算全部最终异常图，也未重新执行最终评估。','',
        'provenance.json 保存本轮源码和实际消费文件哈希；原正式报告和验收记录不变。后续视觉解释见 interpretation.md。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    outputs={p.relative_to(OUT).as_posix():file_hash(p) for p in OUT.iterdir() if p.is_file() and p.name!='provenance.json'}
    save_json(OUT/'provenance.json',dict(created_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),protocol_sha256=DIGEST,
                                       inputs_sha256=inputs,outputs_sha256=outputs,verification='Consumed files verified against accepted integrity manifests; full matrix audit preserved separately'))
    print(json.dumps(overview),flush=True)


if __name__=='__main__':main()


