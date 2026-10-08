# 项目运行与开发基线 v0.1

## 获取数据

在项目根目录使用 PowerShell：

```powershell
.\.venv\Scripts\python.exe scripts/download_assets.py
.\.venv\Scripts\python.exe scripts/prepare_data.py --verify-existing
```

下载脚本使用 Anomalib 数据模块列出的公开 MVTec AD 压缩包与 SHA-256，完整包约 5.26 GB，仅解压 bottle、metal_nut、tile。数据遵循 CC BY-NC-SA 4.0；数据和权重放在 Git 忽略的 `data/`。来源与校验信息写入 `data/mvtec_ad/download_provenance.json`。清单默认写入 `records/manifests/mvtec_v1.json`，存在时拒绝覆盖。

官方 train/good 约 20% 用作正常校准，其余为训练池；官方 test 按正常/具体缺陷类型约 20% 划为开发集，其余留作最终测试。该划分是自定义协议。所有图像和异常掩码都有 SHA-256；运行时校验已使用文件。

## 运行 bottle 开发试验

```powershell
.\.venv\Scripts\python.exe -m anomaly.run --config configs/bottle_pilot.json --output outputs/bottle_A_seed42_n5
.\.venv\Scripts\python.exe -m anomaly.run --config configs/bottle_pilot.json --method B --output outputs/bottle_B_seed42_n5
```

输出目录必须不存在，防止覆盖结果。运行器只允许 `dev`，直接拒绝 `final_test`。配置为 5 张独立正常原图、1,000 个记忆库向量、224×224；条件为干净、亮度 0.7/1.3、对比度 0.7/1.3，是有限条件的管线试运行。

A 使用原图；B 每张原图另加 3 个增强视图；C 另加 1 个。B/C 同一原图的第一个增强视图相同。修改类别、样本数、种子或扰动时复制 JSON 配置并显式修改。相同种子下，5/10/25 张原图按同一顺序嵌套抽样，全量为扣除正常校准后的完整训练池。

| 输出文件 | 内容 |
|---|---|
| `summary.json` / `metrics.csv` | AUROC、FPR、召回率、计数、退化、资源记录 |
| `scores.csv` | 逐图逐条件原始分数与固定阈值判定 |
| `calibration_scores.csv` / `threshold.json` | 正常校准分数、阈值与实际校准 FPR |
| `sample_lists.json` / `training_views.json` | 原图清单、划分、增强操作和系数 |
| `config.json` / `provenance.json` | 配置、版本、源代码/清单/权重哈希 |
| `model.pt` | 记忆库、选中候选索引、阈值与配置 |
| `maps/<condition>/` | 全部原始异常图 `.npy` 和每种缺陷首张示例 `.png` |

PNG 从左到右是输入图片、热力图叠加、真实掩码。显示尺度固定为同一模型校准阈值的 1.5 倍，各条件共用；评分不做测试批次归一化。量化分析使用原始 `.npy`。

## 当前算法

1. 转 RGB，直接双线性缩放至 224×224，转换为 `[0,1]`。
2. 光照变换在 ImageNet 归一化前执行；对比度中心为每个 RGB 通道的空间均值。
3. 冻结 ImageNet 预训练 ResNet18，设为 eval，提取 layer2、layer3。
4. 各层应用 3×3、stride=1 的局部平均池化；layer3 双线性上采样到 layer2，拼接为每个位置 384 维特征。224 输入有 28×28 个候选位置。
5. 候选特征留在 CPU。使用固定种子的 64 维高斯随机投影做贪心最远点选择；起始点由种子决定，后续选离已选集合最近距离最大的点。
6. 记忆库保留原始 384 维特征，容量严格等于预算；候选不足时报错，不重复填充。不同位置的特征值可以相同，但不重复选候选索引。
7. 原始特征空间计算精确欧氏 1-NN，查询/记忆库分块以控制内存。图像分数为最大 patch 距离；异常图双线性上采样。
8. 干净正常校准分数的 95% 线性经验分位数作为阈值，严格大于判异常。每个模型所有条件共用一个阈值。

## 与论文/框架的边界

这是自主实现的简化 PatchCore 流程。已与固定 Anomalib 2.2.0 完成源码及真实开发数据数值核对：在相同输入、权重和记忆库下，特征完全一致，k=1 图像评分在预设浮点容差内一致。详见 [基线核对报告](../reports/baseline_audit.md)。

当前保留固定 64 维高斯投影采样、最大 patch 图像分数、双线性异常图上采样且无高斯平滑。参考默认使用稀疏投影、k=9 邻域重加权、最近邻异常图上采样加高斯平滑。当前实现不宣称复现论文或默认框架数值，也不与其表格直接比较。

目前不截断候选数量。增强会增加 CPU 缓存与采样成本；5 张试验的资源记录不能代表全量或三种子实验的成本。

## 指标与计时口径

- 图像 AUROC：正常=0、异常=1，同分用平均秩；单一标签集合返回空值。
- 像素 AUROC：汇集所有开发图片在 224×224 的像素，包括正常背景；掩码用最近邻缩放。需结合失败案例判断定位质量。
- 固定阈值同时报告 FPR、召回率及分子/分母。
- 单图计时：3 次预热后，batch=1、float32、GPU 前后同步，包含传输、特征、近邻、上采样及输出到 CPU；不含磁盘读图、PIL 缩放或结果保存。
- 每条件对不同开发图片各测一次，报告中位数及数量。三类开发矩阵另加固定 3 张图片各 10 次重复计时，保存到 `inference_benchmark.json`；早期 bottle pilot 没有该文件。
- 各阶段 GPU 峰值通过 PyTorch allocated/reserved 记录；Windows CPU 记录进程至今峰值工作集，不能视为阶段独立峰值。
- 浮点数在不同设备/版本间仍可能有差异，保留硬件与版本信息。

## 验证

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

覆盖数据划分、嵌套抽样、完整性、固定阈值、AUROC 同分、分块距离、记忆库容量、增强配对和最终测试访问限制。

## 当前状态（2026-10-08）

- [x] 完成 Anomalib 2.2.0 数值核对，保留简化实现边界。
- [x] 三种子 A/B/C 共 90 个开发模型、13 个条件；固定容量 1,000。
- [x] formal_v1 已锁定并完成最终评估、配对统计、资源与失败分析。
- [x] 英文报告 v1 已交付；v2 修正采样术语，增加输入校验与独立输出。
- AU-PRO、PaDiM、Demo 为本轮未纳入的可选项。

本目录为工程修订版，不能用修订源码替换历史冻结实现。复现与归档恢复见 [reproduction_v2.md](reproduction_v2.md)。正式结果仍限于三类、自定义划分和合成光照。

## 来源

- [PatchCore 论文](https://arxiv.org/abs/2106.08265)
- [MVTec AD 官方数据与许可](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)
- [Anomalib 下载地址与校验值](https://github.com/open-edge-platform/anomalib/blob/main/src/anomalib/data/datamodules/image/mvtecad.py)
- [Torchvision ResNet18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)
