"""Figures and Chinese report for paired development stability results."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CATS=['bottle','metal_nut','tile']
BUDGETS=['5','10','25','all']
COLORS={'A':'#2563eb','B':'#d97706','C':'#16a34a'}


def find(rows, **query):
    selected=[r for r in rows if all(r[k]==v for k,v in query.items())]
    if len(selected)!=1:
        raise ValueError(f'Expected one row: {query}; found {len(selected)}')
    return selected[0]


def pm(row, metric, signed=False):
    number=f"{row[metric+'_mean']:+.4f}" if signed else f"{row[metric+'_mean']:.4f}"
    return number+f" ± {row[metric+'_std']:.4f}"


def figures(folder,tables):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(3,3,figsize=(16,12))
    specs=[('clean','image_auroc','Clean image AUROC'),('stress','image_auroc_drop','Mean stress AUROC drop'),('stress','fpr','Mean stress FPR')]
    for col,cat in enumerate(CATS):
        for row,(severity,metric,label) in enumerate(specs):
            ax=axes[row,col]
            for method in ['A','B']:
                for seed in [17,42,2026]:
                    values=[find(tables['severity_by_seed'],category=cat,budget=n,method=method,seed=seed,severity=severity)[metric] for n in BUDGETS]
                    ax.plot(range(4),values,color=COLORS[method],alpha=.22,linewidth=1)
                values=[find(tables['severity_seed_summary'],category=cat,budget=n,method=method,severity=severity) for n in BUDGETS]
                ax.errorbar(np.arange(4)+(-.035 if method=='A' else .035),[v[metric+'_mean'] for v in values],
                            yerr=[v[metric+'_std'] for v in values],label=method,color=COLORS[method],marker='o',capsize=3)
            ax.set_xticks(range(4),BUDGETS)
            ax.set_xlabel('Training originals (categorical spacing)')
            ax.grid(alpha=.18)
            if col==0: ax.set_ylabel(label)
            if row==0: ax.set_title(cat); ax.legend()
    fig.suptitle('Final held-out evaluation | three seeds | thin lines: seeds; bars: sample SD, not CI')
    fig.tight_layout(rect=(0,0,1,.96))
    fig.savefig(folder/'seed_stability.png',dpi=140)
    plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,9))
    for ax,metric,label in zip(axes.ravel(),['image_auroc','pixel_auroc','fpr','recall'],
                              ['Image AUROC (higher is better)','Pixel AUROC (higher is better)','FPR (lower is better)','Recall (higher is better)']):
        for i,severity in enumerate(['clean','mild','moderate','stress']):
            values=[find(tables['macro_severity_paired_seed_summary'],budget=n,comparison='B-A',severity=severity) for n in BUDGETS]
            ax.errorbar(np.arange(4)+(i-1.5)*.07,[v[metric+'_mean'] for v in values],yerr=[v[metric+'_std'] for v in values],label=severity,marker='o',capsize=3)
        ax.axhline(0,color='black',linestyle=':',linewidth=1)
        ax.set_xticks(range(4),BUDGETS)
        ax.set(xlabel='Training originals',ylabel='Paired B - A: '+label)
        ax.grid(alpha=.18)
        ax.legend()
    fig.suptitle('Equal category macro within each seed, then mean ± sample SD across three seeds')
    fig.tight_layout(rect=(0,0,1,.96))
    fig.savefig(folder/'paired_effects.png',dpi=140)
    plt.close(fig)
    fig,axes=plt.subplots(3,3,figsize=(16,12))
    for col,cat in enumerate(CATS):
        for row,(metric,label) in enumerate([('image_auroc','Image AUROC'),('fpr','FPR'),('recall','Recall')]):
            ax=axes[row,col]
            for budget,color in [('5','#2563eb'),('25','#d97706')]:
                values=[find(tables['severity_seed_summary'],category=cat,budget=budget,method=m,severity='all_perturbed') for m in ['A','C','B']]
                ax.errorbar([0,1,3],[v[metric+'_mean'] for v in values],yerr=[v[metric+'_std'] for v in values],label='n='+budget,color=color,marker='o',capsize=3)
            ax.set_xticks([0,1,3],['A: 0','C: 1','B: 3'])
            ax.set_xlabel('Added illumination views per original')
            ax.grid(alpha=.18)
            if col==0: ax.set_ylabel(label)
            if row==0: ax.set_title(cat); ax.legend()
    fig.suptitle('View-count ablation | mean over 12 perturbations per seed | bars: three-seed sample SD')
    fig.tight_layout(rect=(0,0,1,.96))
    fig.savefig(folder/'view_ablation.png',dpi=140)
    plt.close(fig)


def render(folder,tables,overview,protocol):
    figures(folder,tables)
    lines=['# 正式最终测试结果', '',
           f"协议：`{protocol['id']}`；SHA-256：`{overview['protocol_sha256']}`。",'',
           f"共 **90 个已冻结模型、1,170 行条件指标、{overview['score_rows']:,} 次评分**，来自 **{overview['independent_final_images']} 张独立最终原图**。开发集和最终集分开，不是 MVTec 官方完整测试集成绩。",'',
           '本次直接评估协议锁定前已验收的记忆库和干净校准阈值，没有重新建库或选取最优种子。三个种子给出均值与样本标准差（ddof=1），不自动构成统计显著性证据。', '',
           '## 划分与计数', '', '| 类别 | 正常最终图 | 异常最终图 | 正常校准图 | 训练池 |', '|---|---:|---:|---:|---:|']
    for c in CATS:
        r=overview['counts'][c]
        lines.append(f"| {c} | {r['normal']} | {r['anomalous']} | {r['calibration']} | {r['train_pool']} |")
    lines += ['', '每个方法独立使用 5/10/25/全量正常训练原图；C 仅 5/25 张。开发异常标签曾用于方法开发，这一额外信息使用已披露；最终测试标签仅用于本轮冻结模型的指标计算。', '',
              '## 干净条件与样本预算', '', '| 类别 | 预算 | 方法 | 图像 AUROC | 像素 AUROC | FPR | 召回 |', '|---|---|---|---:|---:|---:|---:|']
    for c in CATS:
        for n in BUDGETS:
            for m in ['A','B']+(['C'] if n in ['5','25'] else []):
                r=find(tables['metrics_seed_summary'],category=c,budget=n,method=m,condition='clean')
                lines.append(f'| {c} | {n} | {m} | '+' | '.join(pm(r,k) for k in ['image_auroc','pixel_auroc','fpr','recall'])+' |')
    lines += ['', '## 方法配对差', '',
              '先在同一类别、预算、条件和种子中相减；每个种子内等权平均类别与 12 个扰动，再对三个种子计算均值与标准差。条件等权不代表现实光照出现频率。FPR 负值表示误报减少，其余指标正值表示改善。', '',
              '| 预算 | 对比 | 图像 AUROC 差 | 像素 AUROC 差 | FPR 差 | 召回差 |', '|---|---|---:|---:|---:|---:|']
    for n in BUDGETS:
        for comparison in ['B-A']+(['C-A','B-C'] if n in ['5','25'] else []):
            r=find(tables['macro_severity_paired_seed_summary'],budget=n,comparison=comparison,severity='all_perturbed')
            lines.append(f'| {n} | {comparison} | '+' | '.join(pm(r,k,True) for k in ['image_auroc','pixel_auroc','fpr','recall'])+' |')
    lines += ['', '![Paired effects](paired_effects.png)', '', '## 压力条件与类别差异', '',
              '0.5/1.5 在查看最终结果前已定义为压力条件。以下为四个压力条件的等权平均，不是最差条件；全部轻度、中度与单条件数据均保留。', '',
              '| 类别 | 预算 | B−A 图像 AUROC | B−A 像素 AUROC | B−A FPR | B−A 召回 |', '|---|---|---:|---:|---:|---:|']
    for c in CATS:
        for n in BUDGETS:
            r=find(tables['severity_paired_seed_summary'],category=c,budget=n,comparison='B-A',severity='stress')
            lines.append(f'| {c} | {n} | '+' | '.join(pm(r,k,True) for k in ['image_auroc','pixel_auroc','fpr','recall'])+' |')
    lines += ['', '![Seed stability](seed_stability.png)', '', '## 增强视图消融', '', '![View count ablation](view_ablation.png)', '',
              'A 为原图，C 增加一个视图，B 增加三个视图。三者原图配对、最终容量相同；视图不作为独立原图。完整 C−A 和 B−C 差值与逐种子值保存在表格中。', '',
              '## 样本数量与退化', '',
              '按 5→10→25→全量逐条件检查退化顺序。每方法共 108 条有依赖的类别/种子/扰动序列，不视为 108 个独立重复；全相同单列。', '',
              '| 方法 | 指标 | 全相同 | 退化不增 | 退化不减 | 非单调 |', '|---|---|---:|---:|---:|---:|']
    for method in ['A','B']:
        for metric in ['image_auroc_drop','fpr_increase']:
            selected=[r for r in tables['rq2_monotonicity'] if r['method']==method]
            counts=[sum(r[metric+'_pattern']==p for r in selected) for p in ['all_tied','nonincreasing','nondecreasing','nonmonotone']]
            lines.append(f'| {method} | {metric} | '+' | '.join(map(str,counts))+' |')
    lines += ['', '## 完整性与资源', '',
              f"90 组全部通过协议、冻结模型、样本清单、恒定阈值和输出 SHA-256 核验。独立两两比较复核全部图像 AUROC、误报/检出计数；预先指定的 {overview['pixel_audit_count']} 组像素指标由原排序算法复核。", '',
              f"最终评估子进程累计 **{overview['process_wall_seconds']/60:.2f} 分钟**，输出 **{overview['artifact_bytes']/2**30:.2f} GiB**。复用模型，建库成本来自原已测量记录，不能把本次评估耗时当成重新训练加评估耗时。", '',
              f"最大 CPU 峰值工作集 **{overview['peak_cpu_gib']:.2f} GiB**，Torch 最大分配 **{overview['peak_torch_allocated_mib']:.1f} MiB**、预留 **{overview['peak_torch_reserved_mib']:.1f} MiB**，不代表整卡占用。", '',
              '推理耗时包括传输、特征、近邻、异常图与回传；排除磁盘/PIL。每模型预热三次，固定前三张最终图各测十次，另保留每次实际评分计时。最终入口缓存同一模型内的原始缩放图，像素 AUROC 用数学等价的负类排序与正类搜索计算，已在开发数据验证，不改变分数、图或指标定义。', '',
              '## 边界与文件', '',
              '- 不根据最终结果删除条件、换种子、重设阈值或调整增强。后续新增假设标为探索性。',
              '- 三类、合成光照、自定义划分和少量正常最终图限制外推；标准差不是置信区间。AU-PRO 未纳入本轮核心指标。',
              '- metrics.csv 保留每类/种子/条件的 FP、正常数、TP、异常数；paired_by_seed.csv 保留配对差。',
              '- severity*、macro* 和 *_seed_summary.csv 提供严重程度、类别宏平均与三种子统计。',
              '- resources.csv、overview.json、pixel_audit.json 和 provenance.json 保存资源、验收与可追溯记录。',
              '- 协议：records/formal_v1/protocol.json；运行状态和全部分数/异常图：outputs/formal_v1/。', '']
    (folder/'report.md').write_text('\n'.join(lines),encoding='utf-8')
