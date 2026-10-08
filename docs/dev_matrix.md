# 三类开发矩阵与资源测量

本阶段只运行开发集。沿用已核对的简化 PatchCore，固定 ResNet18、224×224、layer2/layer3、1,000 个记忆库向量、batch=1、float32。

## 执行范围

- 类别：bottle、metal_nut、tile。
- 训练原图：5、10、25、扣除正常校准后的全量。
- 方法：A 原图、B 每张原图另加 3 个光照增强视图。
- 种子：42。当前为单种子的规模验证，不作为三种子的稳定性证据。
- 测试：干净，以及亮度/对比度各自取 0.9、1.1、0.7、1.3、0.5、1.5，共 13 个条件。
- 总计 24 次建库、312 行条件指标；异常图与逐图分数全部保存。

## 运行与继续

项目根目录 PowerShell：

```powershell
.\.venv\Scripts\python.exe scripts/run_dev_matrix.py
```

程序顺序启动独立子进程，每组的 CPU 峰值独立计量。再次运行同一命令会校验并跳过已完成组；未完成组从头重跑到新的 attempt 目录，保留原失败产物，不做文件覆盖或组内断点续算。

运行计划、数据清单/权重哈希和核心源码哈希在首次执行时固定。改变配置、核心代码或数据清单后必须改矩阵名称开启新的批次，避免混用结果。已有源码快照保存于 `records/dev_matrix_v1/source/`，完整配置在 `plan.json` 和 `configs/`。

状态文件为 `outputs/dev_matrix_v1/status.json`；每组有日志。`RUNNING.lock` 防止同一批次被重复启动。若机器意外退出留下锁，需核实该 PID 的进程已结束后再移除锁，不应在运行中删除。

## 汇总、图表与扰动检查

```powershell
.\.venv\Scripts\python.exe scripts/inspect_dev_illumination.py
.\.venv-audit\Scripts\python.exe scripts/summarize_dev_matrix.py
```

汇总用独立核对环境中的 Matplotlib 绘图，不修改主环境。默认矩阵参数包含在脚本中；其他矩阵名称通过汇总的 `--matrix-name` 指定，当前汇总器只支持每个组合一个种子。

汇总前逐组验证：配置、核心实现、数据清单、样本配对、嵌套抽样、记忆库容量、13 个条件、固定阈值、逐图评分唯一性和异常图存在性。

## 计量口径

- 建库：特征提取与候选缓存阶段，加贪心采样阶段；包含该阶段读图/校验成本，不含骨干初始化。
- 进程墙钟：包含 Python 启动、模型加载、建库、校准、开发评估及保存结果；独立记录每组总耗时。
- GPU：PyTorch 最大 allocated/reserved；不包括驱动、桌面和其他程序。不能当作整卡占用。
- CPU：Windows 进程至今的 PeakWorkingSetSize；每组独立进程，不当作阶段独立峰值。
- 推理基准：3 次预热，然后取固定 3 张开发图，各重复 10 次；GPU 前后同步。包含 CPU→GPU、模型、近邻、异常图及回传；排除磁盘读图、缩放和保存。
- 正式成本预测：A/B 72 次建库 + C（5、25 张）18 次；评估按最终/开发图片数量放大，C 按视图数在 A/B 之间插值。只读最终划分的数量，未加载最终图像进行模型评估。
- 额外 50% 时间为规划余量，不是测量置信区间；不含人工分析、下载、AU-PRO 和后续方法变化。

## 光照检查

`clipping.csv` 使用扰动后、clip 前的浮点 RGB，统计有任一通道超出 `[0,1]` 的像素比例及缺陷区域内的比例。原图本来等于 0 或 1 的像素不自动计作被裁剪。

接触表选每类每种缺陷按文件名排序的首张开发图片，展示 clean、亮度/对比度 0.5 和 1.5 及掩码。它用于人工检查；裁剪比例不是缺陷可见性的充分判据。所有 13 个条件均保留，尚未依据结果删除任何条件。

## 阅读结果

主报告：`reports/dev_matrix_v1/report.md`。优先查看干净代价、同条件 A/B 配对差值及最差光照条件，再查看宏平均。正常开发图只有 4、4、7 张，解释 FPR 时同时查看计数。

当前不计算跨种子标准差或显著性，不将开发结果标为正式测试结果。扩展到更多种子和 C 消融后，再确认正式协议。

## 固定阈值诊断与下一阶段

当前工作目录为 `D:\ChatGPT\111`。基于现有产物复核阈值、分数和误报：

```powershell
Set-Location D:\ChatGPT\111
.\.venv-audit\Scripts\python.exe scripts/diagnose_dev_thresholds.py
```

输出至 `reports/dev_matrix_v1/threshold_diagnostics/`。先核对原始校准分数、样本身份、严格大于规则及全部指标，再生成表和图；不访问最终图像或执行推理。交叉阈值仅作描述性数值检查，不是另选阈值的评测。

下一阶段配置已准备，尚未执行：

```powershell
.\.venv\Scripts\python.exe scripts/run_dev_matrix.py --matrix configs/dev_ab_extra_seeds_v1.json
.\.venv\Scripts\python.exe scripts/run_dev_matrix.py --matrix configs/dev_c_ablation_v1.json
```

前者为三类 × 四档预算 × A/B × seed=17/2026，共 48 组；后者为三类 × 5/25 张 × C × seed=42/17/2026，共 18 组。保持原实现、全部 13 个条件和固定 1,000 容量。加上已有 seed42 A/B，共 90 个开发模型。种子提前记录于 `records/dev_followup_plan.json`，后续需要支持多种子和 C 的新汇总器；现有单种子汇总器不适用。

## 三种子开发阶段已完成

2026-10-07 已完成上述 66 组补充实验的汇总验收，合计 90 个开发模型。前文“尚未执行”为准备阶段的历史状态。多种子与 C 应使用 `scripts/summarize_dev_stability.py`；详见 `docs/dev_stability.md` 与 `reports/dev_stability_v1/findings.md`。
