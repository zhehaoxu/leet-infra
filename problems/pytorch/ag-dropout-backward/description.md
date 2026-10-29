# Dropout 反向：掩码散射与训练/推理的不对称

## 题目描述

inverted dropout 的缩放同时出现在前向与反向：前向是 \(y=x\cdot mask/(1-p)\)，对 \(x\) 求导又得到同一个因子，于是 \(dx=dY\cdot mask/(1-p)\)。听起来只有一行，但真正的三个坑都在约定上：\(1-p\) 的除法**只在训练态做**、`train=False` 时前向是恒等所以反向也必须原样返回 `dY`（此时 `mask` 与 `p` 全被忽略），以及 `mask` 允许是**可广播**的粗粒度掩码（`dropout2d` 就是 `(1,C,1,1)` 这种整通道共享的形状）。本题给你 `dY, mask, p, train`，返回 `dx`。

## 规则

- 完整签名与参数含义只写在 starter 的 docstring 里，那是唯一判题口径。
- `train=True`：\(dx=dY\cdot mask/(1-p)\)，被丢弃位置梯度为 0，保留位置放大 \(1/(1-p)\)。
- `train=False`：直接返回 `dY` 的副本，不使用 `mask`、不除 \((1-p)\)，也不校验 `p` 是否可用。
- `mask` 必须与 `dY` 同形或可广播到 `dY` 形状；训练态下不可广播时抛 ValueError。
- `p` 必须落在 \([0,1)\)：`p=1` 会让 \(1-p=0\)，与负数一样抛 ValueError。

## 示例

`p=0.5`（放大 2 倍），数值取自 test_cases.py 的 EXAMPLE_* 常量：

```
dY   = [[1, 2],     mask = [[1, 0],     dx = [[2, 0],
       [3, 4]]              [0, 1]]            [0, 8]]
```

\(dx=dY\cdot mask/0.5\)：位置 (0,0) 得 \(1/0.5=2\)、(1,1) 得 \(4/0.5=8\)，被丢弃的两格为 0。

## 约束

- `mask` 可为 bool、int 或 float 张量；运算在 float64 下进行，返回 dtype 与 `dY` 一致
- 与 `F.dropout` + `torch.autograd.grad` 在相同掩码下对拍；另用 \(E[mask]/(1-p)=1\) 验证期望保持
