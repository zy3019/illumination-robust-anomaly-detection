"""第一课：先预测 shape，再运行，最后修改代码验证自己的理解。"""

import torch


def main():
    torch.manual_seed(42)

    # Tensor 是多维数组；shape、dtype、device 是首先要检查的属性。
    x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    print("x =", x)
    print("shape:", x.shape, "dtype:", x.dtype, "device:", x.device)
    print("x[0] shape:", x[0].shape)
    print("x[:, 0, :] shape:", x[:, 0, :].shape)
    print("reshape(2, -1) shape:", x.reshape(2, -1).shape)

    # 广播：把长度为 4 的向量加到最后一个维度。
    offset = torch.tensor([10., 20., 30., 40.])
    print("broadcast first row:", (x + offset)[0, 0])

    # 模拟两张 RGB 图片。这里只学习维度，不下载真实图片。
    images = torch.rand(2, 3, 8, 8)
    print("images [batch, channels, height, width]:", images.shape)
    print("first image:", images[0].shape)
    print("first image, keep batch dimension:", images[0:1].shape)
    print("per-image, per-channel mean:", images.mean(dim=(2, 3)).shape)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images_on_device = images.to(device)
    brighter = (images_on_device * 1.1).clamp(0, 1)
    print("calculation device:", brighter.device)
    print("brightness range:", brighter.min().item(), brighter.max().item())
    print("back to CPU:", brighter.cpu().device)

    # 动手练习（自行在这里补充代码）：
    # 1. 创建 shape=(4, 3, 16, 16) 的随机图片 Tensor。
    # 2. 取第 2 张图，分别得到 (3, 16, 16) 和 (1, 3, 16, 16)。
    # 3. 计算每张图每个通道的均值，预期 shape=(4, 3)。
    # 4. 亮度乘以 0.7，并检查 shape 是否改变、数值是否仍在 [0, 1]。


if __name__ == "__main__":
    main()
