# 项目技能

仓库内 `.agents/skills/` 提供 Matt Pocock 的 engineering、productivity 两组共 27 个技能。它们是随代码版本保存的文件；新克隆项目包含全部技能和配套资料。

请在本仓库根目录启动项目会话。直接在聊天中使用 `$技能名` 加任务，例如：

```text
$ask-matt 我想修复一个指标问题，应该用哪些技能？请用中文说明。
$grill-with-docs 帮我明确这个新功能的输入、输出和验收标准。
$tdd 根据已确认的需求，先写失败测试再实现。
$retro 回顾当前开发过程，提出工作环境改进建议。
```

默认项目配置已记录为 GitHub Issues、五种默认分诊标签、单一领域上下文；无需重复 setup。切换任务系统或文档布局时，再运行 `$setup-matt-pocock-skills` 或编辑对应配置。

- [任务系统](issue-tracker.md)
- [标签映射](triage-labels.md)
- [术语和设计决策](domain.md)
- [完整来源与版本清单](../../.agents/skills/mattpocock-install.json)
- [上游许可证](../../.agents/skills/mattpocock-LICENSE.txt)

来源：<https://github.com/mattpocock/skills>，固定提交 `49dd158d1076134a641b33efb035946536778336`。本次未更改上游技能内容，仅将安装清单的目标目录改为项目相对路径。

这套文件不会自动更新。更新时先检查本地定制，比较上游差异，保留许可证并更新来源清单，再作为正常项目改动提交。若同时使用全局安装的同名技能，检查技能选择器显示的文件路径，优先选择本仓库版本。
