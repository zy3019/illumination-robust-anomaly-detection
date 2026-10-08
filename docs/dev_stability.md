# 三种子开发稳定性与增强视图消融

工作目录：`D:\ChatGPT\111`。本阶段只使用既定开发划分，最终测试入口仍关闭。

## 实验范围

- 复用 `dev_matrix_v1` 已完成的 seed42 A/B 24 组。
- `dev_ab_extra_seeds_v1`：seed17/2026，三类、四档原图预算、A/B，共 48 组。
- `dev_c_ablation_v1`：seed42/17/2026，三类、5/25 张、C，共 18 组。
- 合计 90 个开发模型；方法 C 每张原图增加一个视图，B 增加三个，原图与第一个增强视图严格配对。
- 配置和种子在本阶段运行前保存于 `records/dev_followup_plan.json`。该文件保留历史的 prepared 状态，完成状态以各矩阵的 status 和汇总 overview 为准。

## 执行和恢复

```powershell
Set-Location D:\ChatGPT\111
.\.venv\Scripts\python.exe scripts/run_dev_matrix.py --matrix configs/dev_ab_extra_seeds_v1.json
.\.venv\Scripts\python.exe scripts/run_dev_matrix.py --matrix configs/dev_c_ablation_v1.json
.\.venv-audit\Scripts\python.exe scripts/summarize_dev_stability.py
```

两个矩阵顺序执行，避免 GPU 相互争用。批次脚本对已完成组重新校验后跳过，失败组保留产物并写到新的 attempt；不要同时重复启动同一矩阵。汇总器遇到未完成矩阵会拒绝输出最终汇总。

## 验收

- 三个矩阵的核心源码、权重、数据清单及公共配置一致。
- 合并结果必须恰好覆盖预定的 90 组，不允许遗漏、额外组或重复种子。
- 样本预算嵌套；方法间原图完全配对；全量原图集合跨种子相同；校准与开发样本固定。
- 按已记录种子重建训练增强元数据，核对 C 与 B 首视图一致。
- 加载每个保存模型，检查 1,000 × 384 的 float32 记忆库、有限值、1,000 个不重复且未越界的采样索引，以及配置/阈值/来源记录。
- 从校准分数重算 95% 线性分位数；从逐图分数独立重算图像 AUROC、误报、检出、退化量；每个模型在 13 个条件下保持自己的阈值不变。
- 像素 AUROC 沿用已验证评估实现的保存值；本轮不重新读取和排序全部异常图像素。

## 统计口径

每类别、预算、条件分别给出三个种子的均值和样本标准差（ddof=1）。方法差异先在同种子中做 B−A、C−A、B−C，再计算均值与标准差，不能用两组标准差相减代替配对波动。

类别宏平均先在每个种子内等权平均三个类别，然后对三个种子的宏平均值求标准差。严重程度汇总先在每类别/种子内等权平均所属光照条件；四个轻度、四个中度、四个压力条件分别报告，同时保留全部 12 个扰动平均及干净条件。

不把多个光照版本当独立样本，不把 3 类 × 3 种子当成 9 个独立重复；标准差不是置信区间。类别、条件和种子明细均保留在报告 CSV 中。正常开发图仍为 4/4/7 张，结合原始误报/检出计数解读。

## 资源与产物

产物保存于 `reports/dev_stability_v1/`，包括原始条件指标、配对差、逐种子宏平均、跨种子摘要、资源表、图、报告以及输入 SHA-256。C 成本改用真实测量；正式规模估算只放大评估阶段，仍不是最终测试实测。

分析代码为 `development_statistics.py`、`summarize_dev_stability.py` 和 `render_dev_stability.py`。统计测试覆盖配对相关性、缺失/重复种子、缺失消融、宏平均顺序及光照分组完整性。

## D 盘临时目录运行测试

2026-10-07 验收完成：90 个模型，13 项测试通过。若沙箱默认 C 盘临时目录拒绝访问，将本次进程的临时文件放到 D 盘：

```powershell
New-Item -ItemType Directory -Path D:\ChatGPT\111\data\cache\test-temp -Force | Out-Null
$env:TEMP = 'D:\ChatGPT\111\data\cache\test-temp'
$env:TMP = $env:TEMP
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

最终测试尚未运行。开发结果解读见 `reports/dev_stability_v1/findings.md`。
