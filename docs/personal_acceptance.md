# 本人独立讲解与重跑验收

状态：**待本人完成**。AI 已执行的复现和测试，只能证明工程流程可运行，不能证明你独立掌握。无需一次全对；不确定的部分写出来，再针对性补学。

## 讲解题（先不看答案）

1. 用 3–5 分钟说明从输入图到判为异常的全过程。哪些参数被冻结，哪些数据参与建库，阈值来自哪里？
2. 为什么 AUROC=1.0 仍可能把全部正常图判为异常？用一个保存的 FP 案例说明分数和阈值的关系。
3. 5 张训练图之外用了多少校准和开发数据？为什么不能把它表述为全流程只用了 5 张图？
4. A/B/C 的差别是什么？固定容量控制了什么，没有控制什么？候选索引不同是否保证向量数值不同？
5. 为什么 98,280 条评分不是 98,280 个独立样本？三个种子的标准差能说明什么，不能说明什么？
6. 一句话分别回答 RQ1–RQ3，并给出一个收益和一个代价。不能只说增强有效。
7. 本轮指标形状缺陷是怎样产生的？现有正式结果为什么没有据此被判无效？
8. 新增强方法还能在当前最终集上确认吗？如果继续做，应怎样标识和验证？

将自己的回答写入 `records/personal_acceptance/answers.md`，注明实际日期、是否查阅资料、哪些步骤接受了帮助。不要把 AI 提供的文字原样作为独立回答。

## 亲自操作

使用已经演练通过的 `D:\ChatGPT\111_cleanroom`，亲自在终端执行下列命令。输出名单独使用 personal，避免把 AI 的运行当作自己的运行。

```powershell
Set-Location D:\ChatGPT\111_cleanroom
.\.venv\Scripts\python.exe -B -m anomaly.run --config records/dev_matrix_v1/configs/bottle_n5_A_seed42.json --output outputs/personal_bottle_n5_A_seed42
.\.venv\Scripts\python.exe scripts/compare_reproduction.py --actual outputs/personal_bottle_n5_A_seed42 --reference ../111_frozen_restore/outputs/dev_matrix_v1/bottle_n5_A_seed42_attempt01 --receipt records/personal_acceptance/reproduction.json
```

输出目录必须不存在；若尝试失败，保留目录并改用 `_attempt02`。不要删除失败记录或手动把比较结果改成 passed。完成后给助手发送比较结果、你的一段讲解，以及遇到的问题；不需要分享整个数据集。

## 手工复核与通过标准

从 `reports/formal_v1/failure_analysis/selected_cases.csv` 选择一个 FP 和一个 FN，根据 `score > threshold` 手工判断，并解释对应的 clean→扰动变化与 A→B 变化是否相同。写明文件、样本、方法、条件和分母。

- 工程验收：你产生的新输出通过比较脚本；命令、结果和问题记录可追溯。
- 理解验收：能解释特征/记忆库/阈值、划分隔离、配对统计、研究边界，关键概念无误。
- 贡献验收：明确区分本人决策、本人独立实现/修改、AI 辅助实现与执行。
- 若尚不能解释：状态保持“待补充”，不伪造“独立完成”。

助手会根据实际回答给出逐题反馈；你未回答的项目不会自动标记通过。
