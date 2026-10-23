# 分组卷积（Grouped Convolution）

## 题目描述

分组卷积把输入通道分成 `groups` 段，每一段**只在自己的输入通道子集上**卷积、产出
`Cout//groups` 个输出通道，组与组之间彼此不相连。它把参数量和计算量同比例缩小，
是 ResNeXt、MobileNet（depthwise 即 `groups=C`）等网络的关键构件。

## 规则

- 实现 `solve(x, weight, bias, groups)`，`x` 形状 `(N, Cin, H, W)`。
- `weight` 形状 `(Cout, Cin//groups, kh, kw)`（比标准卷积少一个 `groups` 的输入通道维）。
- `stride=1`、`padding=0`；输出形状 `(N, Cout, H-kh+1, W-kw+1)`；输出通道按组连续切分。
- `Cin`、`Cout` 都能被 `groups` 整除；第 `g` 组用输入通道 `[g*(Cin//groups):(g+1)*(Cin//groups)]`
  生成输出通道 `[g*(Cout//groups):(g+1)*(Cout//groups)]`。

## 示例

```
输入：x 为 (1,4,3,3)，通道 0/1/2/3 分别整片填 1/2/3/4
     weight=全 1，形状 (4,2,2,2)；bias=0；groups=2
   group0（通道 0,1）：每个 2x2 窗口求和 = 4*(1+2) = 12 → 输出通道 0,1
   group1（通道 2,3）：每个 2x2 窗口求和 = 4*(3+4) = 28 → 输出通道 2,3
输出：out[0]=out[1]=[[12,12],[12,12]], out[2]=out[3]=[[28,28],[28,28]]
```

## 约束

- float32 输入；`groups >= 1`，`groups` 整除 `Cin` 与 `Cout`
- `groups == 1` 时应与标准卷积一致；`groups == Cin` 时为深度卷积（depthwise）
