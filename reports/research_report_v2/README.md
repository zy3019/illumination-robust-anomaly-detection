# 英文报告修订版 v2

使用原 formal_v1 数值与已验收图表，只修正“不同候选索引”的术语及复现入口。v1 文件保留。

PDF：`output/pdf/illumination_robustness_research_report_v2.pdf`。正文生成源为 `scripts/build_research_report.py`，不要只修改导出的 Markdown。

在项目根目录的新文档环境运行：

```powershell
py -3.14 -m venv .venv-docs
.\.venv-docs\Scripts\python.exe -m pip install -r requirements-docs.txt
.\.venv-docs\Scripts\python.exe scripts/report_integrity.py
.\.venv-docs\Scripts\python.exe scripts/build_research_report.py
```

40 个接受输入由 `records/report_inputs_v1.json` 固定；任何输入变化会在写出新报告前报错。细节见 `docs/reproduction_v2.md`。本次在新安装的 Python 3.14 文档环境实际生成并逐页检查 8 页 PDF；这不替代个人独立讲解验收。
