# 最终测试后的失败案例分析

这是 formal_v1 验收之后的描述性分析。读取既有分数和异常图，不重新推理、不修改模型或阈值；不能将案例解释当成新的确认性实验。

已核对全部 90 个模型、98,280 条评分和 39,312 条 A/B 配对比较；误报事件 2,368 次，漏检事件 4,667 次。它们来自 252 张独立原图的重复评估，不能作为独立错误样本量或部署错误率。

## 完整覆盖与选择规则

下表的不同图像数是各类别/方法内部的并集。A/B 有四种预算，C 只有两种预算，因此不能直接用累计事件数给方法排名。完整逐条件分母见 defect_strata.csv；配对纠正和新增错误见 paired_transitions.csv。

| 类别 | 方法 | 评分次数 | 误报次数 | 漏检次数 | 曾误报的不同正常图 | 曾漏检的不同异常图 |
|---|---|---:|---:|---:|---:|---:|
| bottle | A | 10452 | 203 | 371 | 16 | 8 |
| bottle | B | 10452 | 77 | 369 | 16 | 8 |
| bottle | C | 5226 | 22 | 216 | 11 | 8 |
| metal_nut | A | 14352 | 216 | 1086 | 15 | 31 |
| metal_nut | B | 14352 | 261 | 927 | 15 | 29 |
| metal_nut | C | 7176 | 105 | 590 | 13 | 27 |
| tile | A | 14508 | 652 | 471 | 26 | 18 |
| tile | B | 14508 | 572 | 399 | 25 | 16 |
| tile | C | 7254 | 260 | 238 | 21 | 18 |

每类别分别选 A/B 均误报、B 纠正误报、B 新增误报、A/B 均漏检、B 纠正漏检、B 新增漏检。首先优先三个种子中重复出现的相同模式，再优先压力条件；误报优先全量预算，漏检优先小预算；最后按冻结条件顺序、文件名和种子排序。该规则在最终结果已可见后制定，不是预注册规则。空类别不补选其他角色。

热力图使用保存的异常图除以各自图像阈值，同一面板共用色标。颜色不是概率，图像阈值不是像素分割阈值；插值后的异常图最大值也不必等于原始 patch 最大图像分数。青色轮廓是缺陷标注。

## 案例索引

| 案例 | 类型 | 预算 / 种子 | 条件 | 同模式种子 | A 比值 | B 比值 |
|---|---|---|---|---:|---:|---:|
| [bottle_persistent_fp](bottle_persistent_fp.png) | A/B均误报 | all / 17 | contrast_0.5 | 3/3 | 1.137329 | 1.055911 |
| [bottle_rescued_fp](bottle_rescued_fp.png) | B纠正A误报 | all / 17 | brightness_0.5 | 3/3 | 1.142417 | 0.910181 |
| [bottle_introduced_fp](bottle_introduced_fp.png) | B新增误报 | all / 42 | contrast_1.3 | 2/3 | 0.992761 | 1.002354 |
| [bottle_persistent_fn](bottle_persistent_fn.png) | A/B均漏检 | 5 / 17 | brightness_0.5 | 3/3 | 0.893273 | 0.909030 |
| [bottle_rescued_fn](bottle_rescued_fn.png) | B纠正A漏检 | 5 / 17 | brightness_0.5 | 2/3 | 0.977142 | 1.000327 |
| [bottle_introduced_fn](bottle_introduced_fn.png) | B新增漏检 | 10 / 42 | brightness_0.5 | 2/3 | 1.014401 | 0.978306 |
| [metal_nut_persistent_fp](metal_nut_persistent_fp.png) | A/B均误报 | all / 17 | brightness_0.5 | 3/3 | 1.001435 | 1.033961 |
| [metal_nut_rescued_fp](metal_nut_rescued_fp.png) | B纠正A误报 | 10 / 17 | brightness_1.5 | 2/3 | 1.025278 | 0.914201 |
| [metal_nut_introduced_fp](metal_nut_introduced_fp.png) | B新增误报 | all / 17 | brightness_1.5 | 2/3 | 0.974562 | 1.024003 |
| [metal_nut_persistent_fn](metal_nut_persistent_fn.png) | A/B均漏检 | 5 / 17 | brightness_0.5 | 3/3 | 0.829985 | 0.843059 |
| [metal_nut_rescued_fn](metal_nut_rescued_fn.png) | B纠正A漏检 | 5 / 17 | brightness_1.3 | 3/3 | 0.992773 | 1.004244 |
| [metal_nut_introduced_fn](metal_nut_introduced_fn.png) | B新增漏检 | 5 / 42 | brightness_0.5 | 2/3 | 1.013428 | 0.980307 |
| [tile_persistent_fp](tile_persistent_fp.png) | A/B均误报 | all / 17 | brightness_0.5 | 3/3 | 1.050680 | 1.024596 |
| [tile_rescued_fp](tile_rescued_fp.png) | B纠正A误报 | all / 17 | brightness_1.5 | 3/3 | 1.045037 | 0.954057 |
| [tile_introduced_fp](tile_introduced_fp.png) | B新增误报 | 10 / 17 | brightness_0.7 | 3/3 | 0.936815 | 1.060185 |
| [tile_persistent_fn](tile_persistent_fn.png) | A/B均漏检 | 5 / 17 | brightness_0.5 | 3/3 | 0.931249 | 0.932726 |
| [tile_rescued_fn](tile_rescued_fn.png) | B纠正A漏检 | 5 / 17 | contrast_0.5 | 3/3 | 0.999647 | 1.005217 |
| [tile_introduced_fn](tile_introduced_fn.png) | B新增漏检 | 25 / 17 | brightness_1.5 | 2/3 | 1.017577 | 0.960810 |

每个案例的原图路径、候选数、缺陷面积、裁剪比例保存在 selected_cases.csv。所有候选保存在 paired_cases.csv，所有错误保存在 errors.csv；没有只保留图示案例。

## 可复现与边界

运行 `.\.venv-audit\Scripts\python.exe -B scripts/analyze_final_failures.py`。入口先检查锁定协议与验收状态；对消费的评分、元数据及选中异常图校验原 integrity.json 哈希，对输入图像/掩码校验原清单。未重复重算全部最终异常图，也未重新执行最终评估。

provenance.json 保存本轮源码和实际消费文件哈希；原正式报告和验收记录不变。后续视觉解释见 interpretation.md。
