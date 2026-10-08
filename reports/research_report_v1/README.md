# 研究报告与失败案例交付

整理日期：2026-10-08。正式实验仍为已冻结、已验收的 `formal_v1`；本轮只整理已保存证据，没有更新模型或阈值。

- 最终英文报告：[8 页 PDF](../../output/pdf/illumination_robustness_research_report.pdf)。
- 英文可编辑文本：[report.md](report.md)；结构化排版内容：[report_content.json](report_content.json)。
- 中文案例解读：[interpretation.md](../formal_v1/failure_analysis/interpretation.md)。
- 18 组完整对照图与索引：[案例报告](../formal_v1/failure_analysis/report.md)。
- 个人贡献与工具使用披露：[contributions.md](contributions.md)。
- 正式全部结果：[原始正式报告](../formal_v1/report.md)；本轮文件来源：[provenance.json](provenance.json)。

## 分析范围

核对 90 个模型、98,280 条评分及 39,312 条 A/B 配对比较。全部 7,035 个错误事件均保留，包含 2,368 次误报和 4,667 次漏检。评分来自 252 张独立原图；累计错误事件数不是独立样本量或部署错误率。

三类各展示六种配对模式：A/B 都误报、B 纠正误报、B 新增误报，以及对应的三种漏检模式。选择规则在最终结果可见后制定，属于描述性分析。它优先跨种子重复出现的模式，全部候选仍在表中，不能把图示样本当作随机抽样。

## 复现

在 `D:\ChatGPT\111` 运行，无需重新进行 GPU 推理：

```powershell
.\.venv-audit\Scripts\python.exe -B scripts/analyze_final_failures.py
.\.venv-audit\Scripts\python.exe -B scripts/render_research_case_figures.py
& 'C:\Users\LENOVO\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' scripts/build_research_report.py
```

第一个命令复核协议、已消费的评分/元数据及选中异常图，并生成完整案例图。第二个命令读取同一批已保存数组，生成适合报告版面的案例和统计图。第三个命令使用独立文档运行时的 ReportLab/Pillow/pypdf 排版，避免向已冻结实验环境安装文档依赖。文档运行时位置与包版本见 `render_environment.json`。

报告英文正文由 `scripts/build_research_report.py` 中的内容段落生成，同时导出 Markdown 和 JSON。Markdown 可以手工编辑；若要自动重建一致的 PDF，请同步修改该脚本中的正文，重建会覆盖导出的 Markdown/JSON。案例中文解读为独立编辑文件，不会被分析脚本覆盖。

PDF 逐页渲染检查命令：

```powershell
& 'C:\Users\LENOVO\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\poppler\Library\bin\pdftoppm.exe' -r 100 -png output/pdf/illumination_robustness_research_report.pdf tmp/pdfs/research_report_v1/page
```

分析环境使用现有 `.venv-audit`；Windows 的 `Failed to find real location of C:\Python314\python.exe` 启动提示没有影响本次命令成功。以退出码、实际产物及验收记录为准。

## 研究表述边界

英文报告为项目研究稿，未虚构作者、单位、投稿或评审信息。项目目标、实现产物、AI 辅助执行与个人独立贡献分开表述。正式结果保留三种子标准差、干净性能代价、失败例子和负面结果，不宣称统计显著、官方完整测试成绩或普遍工业可靠性。
