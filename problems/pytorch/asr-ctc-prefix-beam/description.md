# CTC Prefix Beam Search 账

## 题意

CTC 的 beam 侧解码：不是逐帧取 argmax，而是在**前缀空间**上维护 beam——每个前缀 s 记两个概率：以 blank 结尾（p_b）与以非 blank 结尾（p_nb）的路径概率和。本题转写 Graves（1504.08129）prefix beam search 的无 LM 口径：给定逐帧概率矩阵 `[T, V]`，输出按概率降序的前 k 个前缀及其概率。

## 求解

口径题面钉死：

- 每帧对 beam 中每个前缀 s、每个类 c 展开：
  - **blank 类**：`p_b(s) += (p_b(s) + p_nb(s)) * P(blank)`——blank 不改变前缀，且两种结尾状态都可来；
  - **非 blank 类 c，且 s 末尾恰为 c（重复字符）**：加新字符的路径**只能来自 blank 结尾**（`p_nb(s+c) += p_b(s) * P(c)`；否则会与已发射的同一字符合并成重复），重复延续不改前缀（`p_nb(s) += p_nb(s) * P(c)`）；
  - **非 blank 类 c，其余**：`p_nb(s+c) += (p_b(s) + p_nb(s)) * P(c)`。
- 每帧后按总概率 `p_b + p_nb` 降序剪枝到 `beam_width`；**并列按前缀字典序升序**（稳定可复现）。
- 输出过滤零概率假设；`k` 缺省取 `beam_width`。

初始 `beam = {(): (1.0, 0.0)}`（注意：初始划分只在首帧被消费为 p_b+p_nb 之和，随后被首帧重写——`(0,1)` 初始为数学等价变异，已记 notes）。返回 `{"hypotheses": [{"prefix", "prob"}]}`。

## 数据规模

- 1 ≤ T ≤ 10^4，2 ≤ V ≤ 10^5（beam 内前缀数受 beam_width 控制）。
- probs 空/行空/行不齐/负数/NaN/Inf、blank_id 越界、beam_width/k 非正 int 或 bool → `ValueError`。
