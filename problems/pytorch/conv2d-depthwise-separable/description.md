# 深度可分离卷积（Depthwise Separable Convolution）

## 题目描述

深度可分离卷积把一个标准卷积拆成两步：**depthwise**（每个通道各自用一个 $k\times k$ 核，只在空间上卷积）
+ **pointwise**（$1\times1$ 卷积，跨通道线性组合）。它把计算量从 $C_{in}C_{out}k^2$ 降到
$C_{in}k^2 + C_{in}C_{out}$，是 MobileNet / Xception / EfficientNet 的核心。

## 规则

- 实现 `solve(x, dw_weight, pw_weight, dw_bias=None, pw_bias=None)`，`x` 形状 `(N, C, H, W)`。
- **第一步 depthwise**：`dw_weight` 形状 `(C, 1, kh, kh)`（`kh` 为奇数），`stride=1`、
  `padding = kh//2`（保持空间尺寸不变），逐通道各用自己的核；得到 `(N, C, H, W)`，加 `dw_bias`。
- **第二步 pointwise**：`pw_weight` 形状 `(Cout, C, 1, 1)`，做 $1\times1$ 卷积把 `C` 个通道
  线性组合成 `Cout`（无 padding），加 `pw_bias`；输出 `(N, Cout, H, W)`。

## 示例

```
输入：x=全1 (1,2,3,3)；dw_weight=全1 (2,1,3,3)（padding=1）
     pw_weight=[[[0],[1]],[[2],[3]],[[4],[5]]] 形状 (3,2,1,1)
   depthwise 后中心=9（3x3 全 1 求和），角=4、边=6
   pointwise：out0=0*dw0+1*dw1, out1=2*dw0+3*dw1, out2=4*dw0+5*dw1
输出：(1,3,3,3)
  ch0=[[4,6,4],[6,9,6],[4,6,4]]
  ch1=[[20,30,20],[30,45,30],[20,30,20]]
  ch2=[[36,54,36],[54,81,54],[36,54,36]]
```

## 约束

- float32 输入；`kh` 为奇数；`dw_bias`/`pw_bias` 可为 `None`
- 等价于 `F.conv2d(F.conv2d(x, dw_weight, dw_bias, padding=kh//2, groups=C), pw_weight, pw_bias)`
