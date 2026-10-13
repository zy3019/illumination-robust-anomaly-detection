# 任务系统：GitHub

需求、规格和任务保存到 `zy3019/illumination-robust-anomaly-detection` 的 GitHub Issues。无论当前分支或远程名称如何，命令显式指定此仓库。

使用已在当前执行环境登录的 `gh` CLI。下列 `<编号>`、`<标题>`、`<文件>` 等是参数占位符，替换后再执行。

| 操作 | 命令 |
| --- | --- |
| 新建 | `gh issue create --repo zy3019/illumination-robust-anomaly-detection --title "<标题>" --body-file <文件>` |
| 读取 | `gh issue view <编号> --repo zy3019/illumination-robust-anomaly-detection --json number,title,body,labels,comments` |
| 列表 | `gh issue list --repo zy3019/illumination-robust-anomaly-detection --state open --json number,title,body,labels,comments` |
| 评论 | `gh issue comment <编号> --repo zy3019/illumination-robust-anomaly-detection --body-file <文件>` |
| 添加标签 | `gh issue edit <编号> --repo zy3019/illumination-robust-anomaly-detection --add-label <标签>` |
| 移除标签 | `gh issue edit <编号> --repo zy3019/illumination-robust-anomaly-detection --remove-label <标签>` |
| 关闭 | `gh issue close <编号> --repo zy3019/illumination-robust-anomaly-detection` |

多行内容先写入 UTF-8 文件，再使用 `--body-file`，保留真实换行。关闭前记录完成依据。分诊标签采用 [角色映射](triage-labels.md)。

“发布到任务系统”表示创建 GitHub Issue；“获取任务”表示读取对应 Issue 的正文、标签和评论。

## 子任务与阻塞关系

优先使用原生子任务。CLI 支持时，运行 `gh issue edit <父编号> --repo zy3019/illumination-robust-anomaly-detection --add-sub-issue <子编号>`。不支持时，子任务正文写 `Part of #<父编号>`，父任务维护子任务清单。

通过 `gh api repos/zy3019/illumination-robust-anomaly-detection/issues/<编号> --jq .id` 获取 Issue 数据库 ID。创建阻塞关系使用 `gh api --method POST repos/zy3019/illumination-robust-anomaly-detection/issues/<被阻塞编号>/dependencies/blocked_by -F issue_id=<阻塞任务数据库ID>`；数据库 ID 与显示编号分别使用。

原生依赖不可用时，在正文记录 `Blocked by: #<编号>`。所有阻塞任务关闭后，该任务才可执行。

## Wayfinder

- 路线图为一个标有 `wayfinder:map` 的 Issue，保存 Notes、Decisions-so-far、Fog。
- 子任务关联到路线图，按类型使用 `wayfinder:research`、`wayfinder:prototype`、`wayfinder:grilling` 或 `wayfinder:task`；首次使用时创建缺失标签。
- 待执行任务必须未关闭、无未关闭的阻塞任务、未指派；按路线图顺序选择。
- 领取使用 `gh issue edit <编号> --repo zy3019/illumination-robust-anomaly-detection --add-assignee @me`。
- 解决后先追加答案评论，再关闭任务，并在路线图中记录摘要及链接。

## Pull requests as a triage surface

**PRs as a request surface: no.**

GitHub Issues 与 PR 共用编号空间。引用对象不明确时，先识别类型，再执行对应操作。
