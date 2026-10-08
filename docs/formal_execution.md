> 本页保留历史运行说明。v2 的恢复调度器使用独立 `--frozen-root`，当前步骤见 [reproduction_v2.md](reproduction_v2.md)。不要更改旧协议 digest 来适配新源码。

# 正式评估运行与恢复

工作目录：`D:\ChatGPT\111`。正式协议与模型清单已经冻结，本文件只说明操作，不改变协议。

锁定 SHA-256：`58efbb9a2e54a4e276d062235f141f687a0aeaa84f26e48eb34cfe34b82a6885`。

```powershell
Set-Location D:\ChatGPT\111
.\.venv\Scripts\python.exe -B -u scripts/run_final_matrix.py --protocol records/formal_v1/protocol.json --digest 58efbb9a2e54a4e276d062235f141f687a0aeaa84f26e48eb34cfe34b82a6885
.\.venv-audit\Scripts\python.exe -B -u scripts/summarize_final_results.py --protocol records/formal_v1/protocol.json --digest 58efbb9a2e54a4e276d062235f141f687a0aeaa84f26e48eb34cfe34b82a6885
```

状态保存在 `outputs/formal_v1/status.json`，每组有日志、曝光记录、原始评分、异常图及 integrity.json。已完成组重新核验后跳过；未完成组保留并写到新的 attempt。不要同时启动第二个调度器。若异常终止留下 RUNNING.lock，必须先确认记录的进程已结束，才能移除锁并恢复。

协议、锁定源码、依赖或模型文件变化会导致入口在读取最终图片前停止。不要为了继续运行而修改协议哈希；计算错误须保留原记录，说明影响范围后走协议修订。后续代码实验使用独立版本，不能静默替换锁定实现。

最终报告输出目录为 `reports/formal_v1/`；仅全部 90 组通过验收后写入完成报告。首次读取已记录于每组 exposure_started.json，不能将中断后的数据重新描述为未接触的最终集。

原开发报告和资源估算继续保留。正式评估复用全部已验收的 90 个模型与阈值，运行时间仅表示评估，不含再次建库。

## 2026-10-07 完成记录

90 组最终评估和独立汇总审计已全部完成，验收记录为 `records/formal_v1/acceptance.json`。结果解读见 `reports/formal_v1/findings.md`，完整报告见 `reports/formal_v1/report.md`。

曾在第 40 组中断；确认原进程退出后恢复，39 个完成组校验后跳过，原未完成尝试保留，第 40 组使用 attempt02。中断前后评分和指标一致，记录见 `records/formal_v1/resume_score_check.json`。未修改冻结协议或代码。

90 个有效评估子进程累计 46.94 分钟、有效产物 19.22 GiB，不含中断等待、未完成尝试、额外完整性校验和汇总审计；原中断目录仍保留。最终集已经使用，后续新方法须另行标识为探索性。
