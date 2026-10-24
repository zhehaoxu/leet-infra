# GRU 详解

## 两个门、一个候选

GRU 把 LSTM 的输入门/遗忘门/输出门精简为**两个门**：

- **重置门** $r=\sigma(W_r x + U_r h)$：决定「算候选新状态时」要保留多少旧记忆。
  $r\to0$ 相当于忽略历史，直接以当前输入重开。
- **更新门** $z=\sigma(W_z x + U_z h)$：在旧状态 $h$ 与候选 $\tilde n$ 之间做凸插值，
  扮演了 LSTM 里「遗忘 + 输入」两门的合成功能。
- **候选** $\tilde n=\tanh(W_n x + r\odot U_n h)$：注意 $r$ 只作用在**状态项** $U_n h$ 上
  （PyTorch 用 `gi.split` 后 `r * nh` 实现，而非先把 $r$ 乘进整条 $gh$）。

$$ h' = (1-z)\odot \tilde n + z\odot h $$

$z$ 大 → 多保留旧 $h$；$z$ 小 → 多写入候选。这正是「更新门」之名。

## 与 PyTorch 权重布局对齐

`nn.GRUCell.weight_ih` 是 $(3H, I)$，纵向按 $[r; z; n]$ 三段堆叠；`weight_hh` 同理 $(3H,H)$。
所以正确做法是一次算 `gi = x@w_ih.T + b_ih`（得 $(B,3H)$），再 `.split(H, 1)` 切三段，
而不是提前把权重切开——这样两次矩阵乘最省。

## 易错点

- **门顺序**：PyTorch 是 **(reset, update, new)**。若记成 (update, reset, new) 会得错误结果，
  本 dump 阶段专门对拍过 `nn.GRUCell` 以固定顺序。
- **$r$ 的乘法位置**：`tanh(gi_n + r * gh_n)`，$r$ 只乘 $gh$ 的 new 段，不乘 $gi$ 的 new 段。
- **权重元素数**：`w_ih` 是 $3H\times I$、`w_hh` 是 $3H\times H$（dump 曾把 arange 尺寸写错导致 view 失败）。

## 应用

GRU 参数量比 LSTM 少 25%（少一个门），在小数据集/短序列上常持平或更优，广泛用于
语音、时序预测、机器翻译编码器。多个 GRUCell 沿时间步堆叠即 `nn.GRU`；双向/多层在其上包装。
