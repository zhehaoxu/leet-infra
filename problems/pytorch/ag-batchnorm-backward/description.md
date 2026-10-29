# BatchNorm 训练态反向：三段式与双重计入

## 题目描述

BatchNorm 在训练态是**可导的统计量**：\(\mu,\sigma\) 由当前 batch 自己算出，于是每个 \(x_{n,c}\) 既直接通过 \(x̂_{n,c}\) 影响输出，又通过它参与构造的 \(\mu_c,\sigma_c\) 影响同一列的**所有**样本。反向因此天然分三段——直通项、减均值项、除标准差项——而把后两项的 \(1/N\) 因子写成 \(1/N^2\)（或反过来重复计入一次）就是这道题最著名的「双重计入」错误。本题给你 `x (N,C), gamma, beta, dY`，返回 `(dX, dGamma, dBeta)`。

## 规则

- 完整签名与参数含义只写在 starter 的 docstring 里，那是唯一判题口径。
- 归约沿**第 0 轴（batch）**逐特征进行，方差用有偏估计（`unbiased=False`）。
- 记 \(a=dY\cdot\gamma\)，则 \(dX=\big(a-\text{mean}_N(a)-\hat x\cdot\text{mean}_N(a\circ\hat x)\big)/\sigma\)，三个 mean 全部沿 batch 轴。
- \(d\gamma=\sum_N dY\circ\hat x\)、\(d\beta=\sum_N dY\)，形状 `(C,)`。
- 必须用**本批数据自身**的统计量（训练态），不得引入 `running_mean/running_var`；`eps<0`、形状不符一律抛 ValueError。

## 示例

4 个样本、2 个特征、`eps=0.0`，数值取自 test_cases.py 的 EXAMPLE_* 常量：

```
x      = [[-1,  1],     gamma = [2, 1]  beta = [0, 0]   dY = [[1, 2],
          [ 1, -1],                                          [3, 4],
         [-1,  1],                                           [5, 6],
          [ 1, -1]]                                          [7, 8]]
mu = [0, 0], std = [1, 1], a = dY*gamma = [[2, 2], [6, 4], [10, 6], [14, 8]]
mean_N(a) = [8, 5], mean_N(a*xhat) = [2, -1]
dX     = [[-4, -2], [-4, -2], [4, 2], [4, 2]]
dGamma = [4, -4]     dBeta = [16, 20]
```

第 0 列首行：\(2-8-(-1)\cdot2=-4\)。由于 \(\hat x\) 只有 \(\pm1\) 两种取值，同列的 dX 只可能出现两个不同值。

## 约束

- `x` 为 float32 或 float64 的二维张量；内部 float64 累计，返回 dtype 与 `x` 一致
- 与 `torch.nn.functional.batch_norm(train=True)` 的 `torch.autograd.grad` 对拍，float64 atol=1e-9、float32 atol=1e-5
- 自检：`eps=0` 时 \(\sum_N dX=0\) 且 \(\sum_N dX\circ\hat x=0\) 必须精确成立
