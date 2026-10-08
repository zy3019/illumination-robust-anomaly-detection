# 复跑基线核对

主项目 `.venv`、MVTec 数据、预训练权重及 A/B 原始试验需已按 README 准备。下面在项目根目录 PowerShell 执行。

## 独立环境（首次）

```powershell
py -3.14 -m venv .venv-audit
$projectSite = (Resolve-Path -LiteralPath '.venv/Lib/site-packages').Path
Set-Content -LiteralPath '.venv-audit/Lib/site-packages/project_runtime.pth' -Value $projectSite -Encoding ascii
.\.venv-audit\Scripts\python.exe -m pip install -r records/requirements-audit-lock.txt --constraint records/audit-core-constraints.txt --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu128
.\.venv-audit\Scripts\python.exe -m pip check
```

`.pth` 让独立环境复用主项目已有的 PyTorch/NumPy 等包，避免再次下载 GPU 运行时。新增 Anomalib 及依赖安装到 `.venv-audit`，约束文件防止核心版本变化。项目移动位置后需重新写入 `.pth` 的绝对路径。

Anomalib 固定为 2.2.0。Python 3.14 不在该版本发布时列出的 Python 分类器中，本机安装与实际运行已通过，不代表所有平台兼容。包导入触发的视频模块需要 requests，本次补齐并锁定了该依赖；未执行视频算法。

## 固定参考源码与数值核对

```powershell
.\.venv\Scripts\python.exe scripts/fetch_audit_reference.py
.\.venv-audit\Scripts\python.exe scripts/audit_baseline.py --output outputs/baseline_audit_repeat
```

输出目录必须是新目录。脚本检查实际安装的七个参考源文件与固定提交一致，不会仅凭版本号信任源码。

核对器固定读取 `outputs/bottle_A_seed42_n5` 和 `outputs/bottle_B_seed42_n5`，并核实 checkpoint 与配置、权重及数据清单哈希；源图片/掩码也由现有加载器校验。脚本失败时不会写出 `status=passed` 的最终报告。

输出包括 `audit.json`、各组参考采样记忆库、逐图评分、校准分数和所有条件指标。`same_bank_k1` 与当前简化模型的图像分数应在容差内一致，但它的异常图仍使用参考平滑，因此像素指标不要求相同。

本次成功输出为 `outputs/baseline_audit_v2`。第一次运行在复制旧权重的 BatchNorm 计数器时中止，未产生成功报告或模型实验结果；核对脚本已按 Torchvision 的兼容行为补零后完成验证。

完整检查主环境：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```
