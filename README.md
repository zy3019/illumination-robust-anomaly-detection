# 有限正常样本下工业异常检测的光照鲁棒性研究

项目范围与实验协议以 `计算机视觉科研项目_16周计划大纲_修订版.md` 为准。

已直接进入项目开发，基础练习不作为前置要求。目前实现了简化 PatchCore 开发基线：数据划分、特征提取、固定容量记忆库、A/B/C 光照增强、正常校准阈值、开发集评分与热力图。

运行方法与实现边界见 [项目运行说明](docs/implementation.md)。正式协议已锁定为 formal_v1，最终测试通过独立的协议校验入口运行，详见 [正式评估执行说明](docs/formal_execution.md)。

已完成 [Anomalib 2.2.0 基线实现核对](reports/baseline_audit.md)：特征与匹配配置的评分通过数值对照，原 A/B 结果精确复现。当前保留已披露的采样、重加权和异常图处理差异。

三类开发实验：见 [执行与资源计量说明](docs/dev_matrix.md)，运行 `scripts/run_dev_matrix.py`。本轮为 seed=42 的 24 次建库、13 个条件，结果保存至 `reports/dev_matrix_v1/`。

首轮 seed42 的 24 组已完成：[开发与资源报告](reports/dev_matrix_v1/report.md)、[主要发现与强扰动检查](reports/dev_matrix_v1/findings.md)。后续三个种子的 A/B/C 共 90 个开发模型也已验收，累计子进程耗时 38.53 分钟；采用真实 C 成本更新后的正式规模估计约 110 分钟（增加规划余量约 165 分钟），输出约 22.2 GiB。正式最终测试现已完成并验收：90 个模型、252 张独立最终原图、98,280 次评分；有效评估子进程累计 46.94 分钟、产物 19.22 GiB。以上 110/165 分钟为此前估算，正式评估复用冻结模型且采用已验证的执行优化。详见 [最终结果解读](reports/formal_v1/findings.md) 和 [完整正式报告](reports/formal_v1/report.md)。

## 环境

- Windows x64，Python 3.14.4。
- NVIDIA GeForce RTX 4060 Laptop GPU，8188 MiB，驱动 591.86（初次检查值）。
- W1 选择 PyTorch 2.11.0 + Torchvision 0.26.0，CUDA 12.8 预编译包。
- 实际运行验证以 `records/environment.json` 为准。
- 当前项目直接使用 PyTorch、Torchvision、NumPy、Pillow，无需安装新依赖。后续引入 Anomalib 前单独检查兼容性。

## 重建与运行（PowerShell，项目根目录）

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-w1.txt
.\.venv\Scripts\python.exe scripts/check_environment.py --require-cuda
.\.venv\Scripts\python.exe scripts/download_assets.py
.\.venv\Scripts\python.exe scripts/prepare_data.py
.\.venv\Scripts\python.exe -m anomaly.run --config configs/bottle_pilot.json --output outputs/bottle_A_seed42_n5
```

完整依赖快照在成功安装后保存到 `records/requirements-w1-lock.txt`，可用它替换上面的 requirements 文件重建当时环境。
安装方式参考 [PyTorch 官方版本组合](https://pytorch.org/get-started/previous-versions/#v2-11-0)。

## 文件安排

- `anomaly/`：数据、模型、指标、开发试验运行器。
- `configs/`：试验配置。
- `docs/`：项目运行与实现说明。
- `tests/`：数据划分、指标、采样与评分验证。
- `reports/`：开发与正式最终测试报告、结果表和图表。
- `learning/w01/`：练习说明、示例和个人笔记。
- `scripts/check_environment.py`：CPU/GPU 运算验证及环境记录。
- `records/`：可追溯的环境与进度记录。
- `.venv/`：项目独立环境，Git 忽略。
- 后续 `data/`、`outputs/`、`checkpoints/` 存放大文件，Git 忽略。

当前 Git 仓库已存在。开发结果与最终测试结果分开记录，原始数据与大型模型文件不进入版本库。

当前项目目录：`D:\ChatGPT\111`（C 盘旧路径保留兼容链接）。已完成 [固定阈值诊断报告](reports/dev_matrix_v1/threshold_diagnostics/report.md)，复核全部 24 个模型和 6,552 条开发评分。补充两个种子的 A/B 与三个种子的 C 消融已完成，共新增 66 组。合并后 90 个开发模型已核验，见 [三种子结果解读](reports/dev_stability_v1/findings.md) 和 [完整报告](reports/dev_stability_v1/report.md)。[正式协议](docs/formal_protocol_v1.md) 已锁定，90 组最终测试已全部完成并验收；[最终解读](reports/formal_v1/findings.md)、[验收记录](records/formal_v1/acceptance.json)、[执行说明](docs/formal_execution.md)。


2026-10-08：已完成 [失败案例解读](reports/formal_v1/failure_analysis/interpretation.md) 与 [18 组案例索引](reports/formal_v1/failure_analysis/report.md)，并整理 [8 页英文研究报告及复现说明](reports/research_report_v1/README.md)。本轮仅分析既有结果，未改变正式模型、阈值或协议。
