# LSTM 反向与 BPTT：门控拆分与时间线累加

## 题目描述

LSTM 的反向同时踩中两个坑：门向量的**切分顺序**和**沿时间轴的累加**。PyTorch 的 `W_ih` 行块顺序是 `(i, f, g, o)`（input / forget / cell candidate / output），把 forget 与 cell 两段切错，前向几乎看不出问题、反向却全错。时间方向上，\(dh_t\) 必须同时接收「上游给的 \(dH_{t+1}\)」与「本步产生的分量」，而 \(dW_{ih}=\sum_t dgates_t^T x_t\) 是跨时间累加。本题给你一步一份的 `xs, h0, c0, W_ih, W_hh, b_ih, b_hh, dhT, dcT`，返回七个梯度。

## 规则

- 完整签名与参数含义只写在 starter 的 docstring 里，那是唯一判题口径。
- 前向：\(gates=x_tW_{ih}^T+h_{t-1}W_{hh}^T+b_{ih}+b_{hh}\)，\(i,f,o=\sigma(\cdot)\)、\(g=\tanh(\cdot)\) 依次取切片 `[0:J]、[J:2J]、[2J:3J]、[3J:4J]`；\(c_t=f\,c_{t-1}+i\,g\)，\(h_t=o\tanh(c_t)\)。
- 反向：\(dc_t=d c_{t+1}+dh_t\odot o\odot(1-\tanh^2c_t)\)（**那个 \(o\) 因子来自 \(h=o\tanh c\)**）；\(di=dc\,g,dg=dc\,i,df=dc\,c_{t-1},do=dh\tanh c\)；折进 \(dgates\) 时 sigmoid 门乘 \(y(1-y)\)、tanh 门乘 \(1-y^2\)。
- \(db_{ih}=db_{hh}=\sum_t dgates_t\)，都是全长 \(4J\) 且**完全相同**的两段，不要按门切成半长。
- 维度不符（`W_ih` 行数不是 \(4J\)、`xs` 末轴不是 \(E\) 等）抛 ValueError；内部 float64，返回 dtype 随各入参。

## 示例

`T=1、N=1、E=2、Hid=1`、bias 全 0，数值取自 test_cases.py 的 EXAMPLE_* 常量：

```
x0=[1,1]  h0=[1]  c0=[0]  W_ih row2=[-2,-2]  W_hh row2=[4]  dhT=[1]  dcT=[0]
gates = [0, 0, (-2-2)+4, 0] = [0,0,0,0] ⇒ i=f=o=0.5, g=0 ⇒ c1=0, h1=0
dc1 = 0 + 1*0.5*(1-0) = 0.5 ⇒ dg = 0.5*0.5 = 0.25, other three gates = 0
dgates = [0, 0, 0.25, 0]
dxs=[[-0.5,-0.5]]  dW_ih row2=[0.25,0.25]  dW_hh row2=[0.25]  db=[0,0,0.25,0]
dh0=[1.0]   dc0=[0.25]
```

## 约束

- 支持 `T ≥ 1`、`N ≥ 1`、`Hid ≥ 1`；上游梯度全 0 时七个返回值必须全 0
- 与 `nn.LSTM`（把 `weight_ih_l0` 等四个参数 `copy_` 进去）的 `torch.autograd.grad` 对拍，float64 atol=1e-8
