# 修订版的复现与冻结产物恢复

正式数值来自已验收的 formal_v1。本版修复工程缺陷，不把已使用的最终集重新称为未接触数据。历史源码保留在基线标签 `formal-v1-baseline` 和独立冻结包中；默认运行新开发实验，输出到新目录。

## 从干净目录重建一个开发模型

Windows x64 / Python 3.14.4 / CUDA 可用。以下在新复制或新克隆的仓库根目录执行；不用复制旧虚拟环境或配置指向旧 site-packages 的 `.pth`。

```powershell
New-Item -ItemType Directory -Path tmp -Force | Out-Null
$env:TEMP = (Resolve-Path tmp).Path
$env:TMP = $env:TEMP
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r records/requirements-w1-lock.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe scripts/check_environment.py --require-cuda
.\.venv\Scripts\python.exe scripts/download_assets.py
.\.venv\Scripts\python.exe scripts/prepare_data.py --verify-existing
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -B -m anomaly.run --config records/dev_matrix_v1/configs/bottle_n5_A_seed42.json --output outputs/reproduction_bottle_n5_A_seed42
```

下载可由已校验的本地数据缓存替代：把三类原始数据文件实际复制到 `data/mvtec_ad/`，将权重复制到 `data/weights/`，仍执行全清单哈希核验。不能只复制旧模型冒充重新建库。新划分研究才使用不带 `--verify-existing` 的创建命令和新清单名。

本次自动演练目录为 `D:\ChatGPT\111_cleanroom`，以新安装依赖和复制后校验的数据为基础。它验证当前机器上的独立目录重建，不能代表其他操作系统/GPU兼容性。

## 恢复冻结包并比较

交付的 `dist/formal_v1_reproduction_bundle.zip` 包含原 90 个记忆库、阈值、配置、样本、开发分数、依赖清单、权重和冻结源码；不含原始数据和最终异常图。归档旁 `.sha256` 固定整体摘要；它只防误改，不是外部签名。

```powershell
$bundleDigest = (Get-Content dist/formal_v1_reproduction_bundle.zip.sha256).Trim()
.\.venv\Scripts\python.exe scripts/frozen_bundle.py restore --archive dist/formal_v1_reproduction_bundle.zip --sha256 $bundleDigest --target ../111_frozen_restore
.\.venv\Scripts\python.exe scripts/run_final_matrix.py --frozen-root ../111_frozen_restore --digest 58efbb9a2e54a4e276d062235f141f687a0aeaa84f26e48eb34cfe34b82a6885 --verify-only
.\.venv\Scripts\python.exe scripts/compare_reproduction.py --actual outputs/reproduction_bottle_n5_A_seed42 --reference ../111_frozen_restore/outputs/dev_matrix_v1/bottle_n5_A_seed42_attempt01 --receipt records/reproduction_comparison.json
```

恢复目标必须不存在；重复验收时复用已恢复目录并执行 verify-only，不重复 restore。比对要求样本、增强记录、阈值、记忆库张量、采样索引、逐图分数和主要指标完全一致；计时、源代码哈希和序列化文件字节不作为跨运行数值等价标准。不同软硬件环境可能无法满足精确相等；应报告差异，不能放宽验收却仍称“精确复现”。

原始工作区可导出新副本：

```powershell
.\.venv\Scripts\python.exe scripts/frozen_bundle.py export --source ../111 --output dist/formal_v1_copy.zip --digest 58efbb9a2e54a4e276d062235f141f687a0aeaa84f26e48eb34cfe34b82a6885
```

若确需重跑历史最终推理，在恢复目录按原清单准备数据，调用本版 `run_final_matrix.py --frozen-root <目录> --digest <上述值>`。新调度器记录自身版本，子进程仍运行冻结目录的原数值评估器。恢复时完整但状态未落盘的 attempt 会先校验后接纳；无完整性清单才重试，已有清单但内容损坏则停止。丢失的子进程计时保持空值；本版汇总工具也支持 `--frozen-root`，按缺失口径报告成本。此流程是历史复跑，不是新的独立确认研究。

## 重建英文报告

无需 GPU、原模型或整套 Anomalib。单独文档环境的可移植命令：

```powershell
py -3.14 -m venv .venv-docs
.\.venv-docs\Scripts\python.exe -m pip install -r requirements-docs.txt
.\.venv-docs\Scripts\python.exe scripts/report_integrity.py
.\.venv-docs\Scripts\python.exe scripts/build_research_report.py
```

新产物为 `reports/research_report_v2/` 和 `output/pdf/illumination_robustness_research_report_v2.pdf`，不会覆盖 v1。正文唯一生成源为脚本；如修改正文，应重建 Markdown、JSON、PDF并重新视觉检查。源表格/案例/图形由 `records/report_inputs_v1.json` 单独固定；修改表格后重建会被拒绝。新增研究结果须另行验收与建立新输入记录，不能自动刷新旧摘要。

PDF 可用已安装的 Poppler `pdftoppm -r 100 -png <PDF> <前缀>` 渲染。需要明确指定其路径时使用实际安装目录，不依赖某个用户名或 Codex 缓存路径。冻结文档正文、原 v1 报告及旧记录均作为历史材料保留。
