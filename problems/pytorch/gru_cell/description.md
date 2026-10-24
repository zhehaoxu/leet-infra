# GRU 单元（GRUCell）

## 题目描述

门控循环单元（GRU）用两个门——**重置门** $r$ 与**更新门** $z$——控制历史状态如何写入新状态，
比 LSTM 少一个门却常有相当表现，是时序建模的基础构件。本题实现**单个时间步**的 GRU 递推。

## 规则

- 实现 `solve(x, h, w_ih, w_hh, b_ih, b_hh)`，输入 `x` 形状 `(B, input_size)`，隐状态 `h` 形状 `(B, hidden_size)`。
- 权重按 PyTorch `nn.GRUCell` 约定：`w_ih` 形状 `(3*hidden_size, input_size)`，
  `w_hh` 形状 `(3*hidden_size, hidden_size)`，`b_ih` / `b_hh` 形状 `(3*hidden_size,)`；
  三行块顺序为 **reset、update、new**（记作 $r, z, n$）。
- 计算（$\sigma$ 为 sigmoid）：
  $$ gi = x\,w_{ih}^{\top} + b_{ih},\quad gh = h\,w_{hh}^{\top} + b_{hh} $$
  $$ r = \sigma(gi_r + gh_r),\quad z = \sigma(gi_z + gh_z),\quad \tilde n = \tanh(gi_n + r \odot gh_n) $$
  $$ h' = (1 - z)\odot \tilde n + z \odot h $$
- 结果须与 `nn.GRUCell(...)`（权重置为本输入）单步前向一致。

## 示例

```
输入：input_size=2, hidden_size=3
     x = [[0.5, -0.5]],  h = [[0.2, -0.3, 0.1]]
     w_ih = arange(18).reshape(9,2)/20 - 0.4
     w_hh = arange(27).reshape(9,3)/20 - 0.4
     b_ih = b_hh = 0
输出：h' = [[0.084566, -0.161684, 0.035316]]
```

## 约束

- float32 输入；`w_ih`/`w_hh` 的第一维必须是 `3*hidden_size`
- 门块顺序固定为 (reset, update, new)，与 `nn.GRUCell` 一致
- 不使用 `nn.GRUCell` / `nn.GRU` / `F` 里的循环算子；需手写门控公式
