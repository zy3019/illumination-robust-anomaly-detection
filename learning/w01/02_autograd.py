"""第二课：一个参数的一次梯度更新。先完成 Tensor 课再看这里。"""

import torch


def main():
    # 预测 y_hat = w*x，目标 y=1，初始预测为 6。
    x = torch.tensor(3.)
    target = torch.tensor(1.)
    w = torch.tensor(2., requires_grad=True)
    optimizer = torch.optim.SGD([w], lr=0.01)

    optimizer.zero_grad()  # 每一步反传前清除之前积累的梯度。
    prediction = w * x
    loss = (prediction - target).square()
    loss.backward()        # 计算梯度，不会自动更新 w。
    print("before: w=", w.item(), "loss=", loss.item(), "gradient=", w.grad.item())
    # 手算：dL/dw = 2*(3*w-1)*3 = 30。
    optimizer.step()       # w <- w - 0.01*30 = 1.7。
    with torch.no_grad():
        new_loss = (w * x - target).square()
    print("after:  w=", w.item(), "loss=", new_loss.item())

    # 动手练习：先预测，再把 lr 改为 0.1 和 1.0，各运行一次。
    # 记录：损失一定下降吗？backward() 与 step() 分别做了什么？


if __name__ == "__main__":
    main()
