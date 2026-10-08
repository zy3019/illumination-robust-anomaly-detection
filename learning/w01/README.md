# W1：环境与 PyTorch 入门

先完成今天的 45–60 分钟练习。能运行示例只是起点，能解释和修改才算掌握。

## 今天：Tensor 的形状与设备

在项目根目录打开 PowerShell，所有命令使用项目的独立 Python，无需激活环境或更改执行策略。

1. 运行环境检查（约 5 分钟）：

   ```powershell
   .\.venv\Scripts\python.exe scripts/check_environment.py --require-cuda
   ```

   查看 `cuda_available` 是否为 `true`，CPU 与 CUDA 的验证是否均为 `passed`。
   结果保存在 `records/environment.json`。

2. 阅读 `01_tensors.py`（约 10 分钟）。运行前在纸上预测 `x[0]`、`x[:, 0, :]`、`x.reshape(2, -1)` 的形状。

3. 运行并对照预测（约 10 分钟）：

   ```powershell
   .\.venv\Scripts\python.exe learning/w01/01_tensors.py
   ```

4. 在文件末尾的练习位置完成四个小练习（约 20 分钟），再运行检查。

5. 在 `notes.md` 中写下自己的解释（约 5 分钟），把不确定的地方带回聊天讨论。

### 今天必须能解释

- `[2, 3, 8, 8]` 在示例中分别表示什么？
- 为什么 `images[0]` 与 `images[0:1]` 的 shape 不同？
- `reshape` 与 `.to("cuda")` 分别改变什么？
- 对图片做亮度变换为什么不需要改变缺陷掩码的几何位置？

## 本周余下安排

- 第二次：运行 `02_autograd.py`，手算梯度与更新后的参数，比较不同学习率。
- 第三次：巩固索引、广播、reshape 和设备转移；故意制造一个维度不匹配并解释报错。
- 第四次：整理笔记与环境记录，学习 `git status`、`git diff`、`git add`、`git commit`，做首次本地提交。
- W2 再进入 Dataset、DataLoader 和完整 MNIST 训练循环。

## 资料

- [PyTorch Tensor 官方教程](https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html)
- [自动求导官方教程](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)

## 学习验收（本人完成后勾选）

- [ ] 我亲自运行了环境检查，知道项目使用哪一个 Python。
- [ ] 我完成了四个 Tensor 练习，并能解释 shape。
- [ ] 我能手算示例的梯度，并区分求梯度与更新参数。
- [ ] 我知道梯度为何要清零，以及评估时为何常用 no_grad。
- [ ] 我写下本周投入时间、问题和自己的解释。

助手安装环境和验证示例不等于本人已完成学习验收。
