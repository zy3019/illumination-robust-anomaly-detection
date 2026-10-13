# 开发工作流

## 选择工作目录和版本

仓库：<https://github.com/zy3019/illumination-robust-anomaly-detection>。

| Git 入口 | 用途 |
| --- | --- |
| `master` | 持续开发和默认克隆入口，包含已验证 v2 及后续改动 |
| `formal-v1-baseline` | 原研究基线，提交 `037b435`；用于定位历史实现 |
| `repair-v2-validated` | 已验收 v2，提交 `87c0cd3`；后续改动的参照点 |

两个标签保持原提交。后续提交沿 master 推进。原有 `repair-v2` 本地分支可以留作历史参照。

原作者各本地目录的用途以 [复现验收记录](../reports/reproduction_v2.md) 中的表格为准；本机后续开发使用 `111_v2`。表格中的盘符和目录名是历史环境记录，不是克隆仓库的前置条件。

在任意新目录克隆仓库即可开始；`cleanroom` 用于独立复现，`frozen_restore` 用于恢复后的历史执行。模型、原始数据和 dist 归档仍按 [.gitignore](../.gitignore) 管理，Git 标签只保存版本库内的文件。完整历史恢复另见 [复现步骤](reproduction_v2.md)，获取复现包的问题与源码开发入口分开处理。

## 开始工作

```powershell
git status --short
git branch --show-current
git remote -v
```

确认当前目录属于需要修改的克隆。`origin` 应指向上面的 GitHub 仓库；整理过的本机 v2 保留 `baseline-local` 远程作为旧工作区的只读参照。新克隆不需要该远程。

已有本地改动先完成提交或妥善保存。在干净 master 上同步：

```powershell
git switch master
git pull --ff-only origin master
```

需要独立审查的改动可从更新后的 master 创建功能分支并提交 PR。发生分叉时检查双方提交并正常合并，不用强制推送覆盖既有历史。

## 运行与验证

实验环境、数据和依赖使用 [复现步骤](reproduction_v2.md)。正式协议与源码快照保持历史字节；新实验写入新的配置和输出目录。

现有实验环境建立后，代码改动运行：

```powershell
New-Item -ItemType Directory -Path tmp -Force | Out-Null
$env:TEMP = (Resolve-Path tmp).Path
$env:TMP = $env:TEMP
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -B scripts/report_integrity.py
```

本次入口整理没有改变历史模型、阈值、协议、源表格或运行器。只改导航文档时，核对路径和链接以及 `git -c core.whitespace=cr-at-eol diff --check` 即可。仓库固定文件原始字节，这个检查设置将 Windows CRLF 识别为换行，同时检查其他行尾空白。上述检查不等价于重跑全部模型，也不代表个人验收通过。

## 提交与推送

先查看 `git status`、`git diff`，再用 `git add <文件或目录>` 暂存本次目标。提交前查看 `git diff --cached` 和 `git -c core.whitespace=cr-at-eol diff --cached --check`；提交说明记录实际改动。master 上的常规推送是：

```powershell
git push origin master
```

GitHub 认证在实际执行推送的环境中检查。使用 GitHub CLI 时，先确保该环境的 PATH 能找到 `gh`，再运行 `gh auth status --hostname github.com` 和 `gh repo view zy3019/illumination-robust-anomaly-detection`。网页登录和终端认证分别验证；一次环境中的成功不能替代另一个环境的检查。

## 研究与任务入口

需求、规格、任务保存在 GitHub Issues，命令与标签见 [任务系统配置](agents/issue-tracker.md)。项目技能随仓库分发，见 [技能指南](agents/README.md)。

研究结论和工程验收的适用范围引用 [正式协议](formal_protocol_v1.md) 及 [复现验收记录](../reports/reproduction_v2.md)，个人掌握情况依据 [本人验收清单](personal_acceptance.md) 单独记录。
