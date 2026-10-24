# LSTM 单元（单步前向）

## 题目描述

LSTM 用「细胞状态 $c$ + 三道门」解决普通 RNN 的梯度消失：遗忘门决定旧记忆保留多少、
输入门决定新信息写入多少、输出门决定读出多少。实现**一个时间步**的前向，是理解
「门控 = 可微的按位开关」以及后续 GRU / 门控循环 / 甚至注意力门控的基础。

## 规则

- 实现 `solve(x, h_prev, c_prev, w_ih, w_hh, b_ih, b_hh)`，与 `torch.nn.LSTMCell` 对齐：
  - `x` 形状 `(B, input_size)`，`h_prev/c_prev` 形状 `(B, hidden)`；
  - `w_ih` 形状 `(4*hidden, input)`，`w_hh` 形状 `(4*hidden, hidden)`，`b_ih/b_hh` 形状 `(4*hidden,)`。
- 门控：$\text{gate} = xW_{ih}^\top + hW_{hh}^\top + b_{ih} + b_{hh}$，按 **[i, f, g, o]** 顺序切 4 段。
  $$ i=\sigma,\ f=\sigma,\ g=\tanh,\ o=\sigma $$
- 状态更新：$c' = f\odot c + i\odot g$，$h' = o\odot\tanh(c')$。
- 返回 `(h_new, c_new)`，均形状 `(B, hidden)`。

## 示例

```
输入：令 w_ih=w_hh=0、b_ih=b_hh=0（于是四个门预激活全为 0），c_prev = 2
  gate 全 0 → i=f=o=σ(0)=0.5，g=tanh(0)=0
  c' = 0.5*2 + 0.5*0 = 1.0
  h' = 0.5*tanh(1.0) ≈ 0.3808
输出：h_new ≈ 0.3808，c_new = 1.0
```

## 约束

- float32 输入；`hidden = w_hh.shape[1]`
- 需与 `torch.nn.LSTMCell`（用同一组权重）在 `atol=1e-6` 内一致
