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
    fig.suptitle('Development only | three seeds | thin lines: seeds; bars: sample SD, not CI')
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


def render(folder,tables,overview):
    figures(folder,tables)
    lines=['# 三种子开发稳定性与 C 消融', '', '实验完成于 2026-10-06，汇总复核于 2026-10-07。工作目录：`D:\\ChatGPT\\111`。本报告使用开发集，最终测试未运行，正式协议尚未锁定。', '',
           f"本轮新增 **66 次建库**；合并已有 seed42 的 24 组，共 **{overview['model_count']} 个模型、{overview['condition_rows']:,} 行条件指标、{overview['scored_views']:,} 条评分**。独立开发原图仍为 **{overview['independent_dev_images']} 张**。", '',
           '## 设计与核验', '',
           '- A/B：三类 × 5/10/25/全量 × 三种子（17、42、2026），共 72 组。C：三类 × 5/25 张 × 同三种子，共 18 组。',
           '- 所有模型沿用已核对的轻量 PatchCore；骨干、权重、划分、224 输入、评分流程一致，记忆库固定 1,000 向量。',
           '- 核对同种子 A/B/C 原图相同、样本预算嵌套、C 的增强视图与 B 的首个增强视图相同。全量组跨种子原图集合相同，顺序、增强和采样可不同。',
           '- 重算所有干净校准阈值、严格大于判定、误报/检出计数、图像 AUROC 和退化量；图像 AUROC 采用独立两两比较。像素 AUROC 使用已验证流程的保存值，本轮不重新排序全部像素。',
           '- 全部 13 个条件保留；0.5/1.5 为压力条件单独解释。训练增强范围仍为 [0.8,1.2]，每个模型的干净校准阈值固定用于所有条件。', '',
           '## 统计口径', '',
           '先计算同类别、同预算、同条件、同种子的 B−A、C−A 和 B−C。跨种子使用均值与样本标准差（ddof=1）。宏平均在每个种子内先等权平均三个类别，再跨三个种子计算标准差，不能将 3 类 × 3 种子当作 9 个独立重复。', '',
           '轻/中/压力条件各包含亮度和对比度的四个条件；全部扰动平均包含 12 条件。条件等权只描述本实验，不代表实际光照出现概率。误差条是三个种子的样本标准差，不能解释为置信区间或显著性。', '',
           '## A/B 配对结果', '',
           '下表为三个类别、12 个扰动条件的宏平均配对差，先在每个种子内聚合。FPR 负值表示误报减少；AUROC 和召回正值表示改善。', '',
           '| 预算 | 图像 AUROC B−A | 像素 AUROC B−A | FPR B−A | 召回 B−A | 各种子图像 AUROC 差（17 / 42 / 2026） |',
           '|---|---:|---:|---:|---:|---|']
    for budget in BUDGETS:
        r=find(tables['macro_severity_paired_seed_summary'],budget=budget,comparison='B-A',severity='all_perturbed')
        seeds=[find(tables['macro_severity_paired_by_seed'],budget=budget,comparison='B-A',severity='all_perturbed',seed=s)['image_auroc'] for s in [17,42,2026]]
        lines.append(f'| {budget} | '+' | '.join(pm(r,m,True) for m in ['image_auroc','pixel_auroc','fpr','recall'])+' | '+' / '.join(f'{s:+.5f}' for s in seeds)+' |')
    lines+=['','![Paired effects](paired_effects.png)','','## 类别差异与干净代价','','表中各数值均为三种子均值 ± 样本标准差。压力条件平均不等于最差条件；各条件的原始 FP/正常数及 TP/异常数见 metrics.csv。','','| 类别 | 预算 | 干净图像 AUROC B−A | 压力图像 AUROC B−A | 压力 FPR B−A | 压力召回 B−A |','|---|---|---:|---:|---:|---:|']
    for cat in CATS:
        for budget in BUDGETS:
            clean=find(tables['severity_paired_seed_summary'],category=cat,budget=budget,comparison='B-A',severity='clean')
            stress=find(tables['severity_paired_seed_summary'],category=cat,budget=budget,comparison='B-A',severity='stress')
            lines.append(f'| {cat} | {budget} | {pm(clean,"image_auroc",True)} | '+' | '.join(pm(stress,m,True) for m in ['image_auroc','fpr','recall'])+' |')
    lines+=['','![Seed stability](seed_stability.png)','','## C 消融：一个与三个增强视图','','仅在 5 和 25 张进行。C−A 检查增加一个视图；B−C 检查再增加两个视图。以下对三类、12 个扰动等权宏平均后汇总三个种子，不将视图数增加预设为单调改善。','','| 预算 | 比较 | 图像 AUROC 差 | 像素 AUROC 差 | FPR 差 | 召回差 |','|---|---|---:|---:|---:|---:|']
    for budget in ['5','25']:
        for comparison in ['C-A','B-C']:
            r=find(tables['macro_severity_paired_seed_summary'],budget=budget,comparison=comparison,severity='all_perturbed')
            lines.append(f'| {budget} | {comparison} | '+' | '.join(pm(r,m,True) for m in ['image_auroc','pixel_auroc','fpr','recall'])+' |')
    lines+=['','![View ablation](view_ablation.png)','','## 固定阈值案例的种子检查','','以下是上一轮已指出的案例，按相同条件展示全部种子，不从新结果挑选最优种子。','','| 类别 / 条件 | 方法 | 种子 | 误报 / 正常 | 检出 / 异常 | 图像 AUROC |','|---|---|---:|---:|---:|---:|']
    for cat,condition in [('bottle','brightness_0.5'),('bottle','contrast_1.5'),('metal_nut','brightness_1.5'),('tile','brightness_0.5')]:
        for method in ['A','B']:
            for seed in [17,42,2026]:
                r=find(tables['metrics'],category=cat,budget='all',method=method,seed=seed,condition=condition)
                lines.append(f'| {cat} / {condition} | {method} | {seed} | {r["false_positives"]}/{r["normal_images"]} | {r["true_positives"]}/{r["anomalous_images"]} | {r["image_auroc"]:.4f} |')
    lines+=['','## 实测资源与更新估算','',
            f"新增 66 组子进程累计 **{overview['new_process_wall_seconds']/60:.2f} 分钟**，产物 **{overview['new_artifact_bytes']/2**30:.2f} GiB**；全部 90 组累计 **{overview['all_process_wall_seconds']/60:.2f} 分钟**，产物 **{overview['all_artifact_bytes']/2**30:.2f} GiB**。时间不包含本轮汇总与人工检查。",'',
            f"90 组最大 CPU 峰值工作集 **{overview['max_cpu_gib']:.2f} GiB**，Torch 峰值分配 **{overview['max_torch_allocated_mib']:.1f} MiB**、预留 **{overview['max_torch_reserved_mib']:.1f} MiB**。Torch 数字不是整卡占用。",'',
            '建库包含特征提取与采样；推理是每组预热后固定三图各十次的中位数（不含磁盘与 PIL）。各配置只执行一次，各种子也改变样本/增强/采样，所以种子间耗时变化不能当作相同配置的重复测速误差。', '',
            f"使用实际 C 运行成本重新估算 90 组正式规模：时间点估计 **{overview['formal_projected_seconds']/60:.1f} 分钟**，加 50% 规划余量约 **{overview['formal_planning_seconds']/60:.1f} 分钟**，产物约 **{overview['formal_projected_bytes']/2**30:.1f} GiB**。仅按最终/开发图片数量放大评估阶段；未执行最终测试。不同日期、缓存、温度和磁盘状态会影响估算，不含 AU-PRO、失败重跑和人工分析。",'',
            '## 限制与后续','',
            '- 正常开发图每类只有 4、4、7 张，误报率分别以 25%、25%、14.29% 跳变；必须结合原始计数和异常召回解释。',
            '- 三个种子刻画当前训练抽样、增强和采样的变化，不涵盖数据划分不确定性；全量组并非三份不同的训练图集合。',
            '- 同一开发集已用于诊断与方法开发，本轮结果不能替代独立最终评估，不自动构成统计显著性证据。',
            '- 下一步保存正式协议：保留三类、三种子、四档 A/B、限定 C、全部光照条件和固定校准规则；明确压力条件与报告口径后再开放最终测试。', '',
            '## 复跑与产物','', '```powershell',r'Set-Location D:\ChatGPT\111',r'.\.venv\Scripts\python.exe scripts/run_dev_matrix.py --matrix configs/dev_ab_extra_seeds_v1.json',r'.\.venv\Scripts\python.exe scripts/run_dev_matrix.py --matrix configs/dev_c_ablation_v1.json',r'.\.venv-audit\Scripts\python.exe scripts/summarize_dev_stability.py','```','',
            '- metrics.csv / resources.csv：1170 行条件指标与 90 组资源，保留种子及原始误报/检出计数。',
            '- paired_by_seed.csv：936 行 B−A、C−A、B−C 配对差；*_seed_summary.csv：三种子均值与样本标准差。',
            '- severity* / macro*：类别与严重程度汇总，并保留先聚合后的逐种子值供核查。',
            '- overview.json / provenance.json：资源估计、核验口径、输入与分析脚本哈希。',
            '- 原模型与逐图评分保存在 outputs/ 下三个矩阵目录；计划、配置和核心源码快照保存在 records/ 对应目录。','']
    lines += ['## 样本数量与退化的顺序检查', '',
              '按 5→10→25→全量检查同类别、同种子、同方法、同光照条件的退化序列。退化不增符合“更多原图减少退化”的方向；全相同单独列出，不能算作支持。该表统计有依赖的序列，不是独立样本数，也不进行显著性检验。', '',
              '| 方法 | 指标 | 全相同 | 退化不增 | 退化不减 | 非单调 |', '|---|---|---:|---:|---:|---:|']
    for method in ['A','B']:
        for metric,label in [('image_auroc_drop','干净 AUROC − 扰动 AUROC'),('fpr_increase','扰动 FPR − 干净 FPR')]:
            selected=[r for r in tables['rq2_monotonicity'] if r['method']==method]
            counts=[sum(r[metric+'_pattern']==pattern for r in selected) for pattern in ['all_tied','nonincreasing','nondecreasing','nonmonotone']]
            lines.append(f'| {method} | {label} | '+' | '.join(map(str,counts))+' |')
    lines += ['', '原始四档数值见 rq2_monotonicity.csv；全量是实际训练池大小，四档没有被当作等距连续数量拟合斜率。', '']
    lines += ['## C 建库成本实测', '', '下表为特征提取加采样的秒数，三种子均值 ± 样本标准差；种子改变输入与采样，不能当作相同配置的重复测速。', '',
              '| 类别 | 原图预算 | A（原图） | C（加 1 视图） | B（加 3 视图） |', '|---|---|---:|---:|---:|']
    for cat in CATS:
        for budget in ['5','25']:
            values=[find(tables['resources_seed_summary'],category=cat,budget=budget,method=m) for m in ['A','C','B']]
            lines.append(f'| {cat} | {budget} | '+' | '.join(pm(r,'build_seconds') for r in values)+' |')
    lines += ['']
    (folder/'report.md').write_text('\n'.join(lines),encoding='utf-8')

