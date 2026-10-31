# KV Cache：Quest 页级 query-aware 上界打分与预算选页

## 题目描述

decode 阶段每来一个 token 都要把整条 KV cache 读一遍，这是长上下文推理最主要的带宽开销。Quest（2406.10774）的思路是：把 KV cache 切成固定大小的 page，为每页每通道维护 key 的极值；每个 decode step 先用 query 和这堆极值**估一个注意力上界**，只把分数最高的若干 page 送进真正的 attention。

上界是这么来的。对固定的 head 与通道 \(d\)，某页内 key 的取值落在 \([\min_i k_{i,d},\ \max_i k_{i,d}]\) 里，于是这一页对该 query 的贡献不会超过

\[
\sum_{i\in\text{page}} q_d\,k_{i,d}\ \le\ q_d\cdot
\begin{cases}\max_i k_{i,d}, & q_d > 0\\[2pt] \min_i k_{i,d}, & q_d \le 0\end{cases}
\]

把 \(q\) 的符号乘到 key 上、对页内取 amax，就得到「每个通道取哪一端」；再与 \(q\) 做点积，一次矩阵乘即可给全部 page 打分。上游 `quest/tests/test_estimate.py::_ref_cpu_estimate` 就是这段 CPU 参考实现，本题逐行照抄它的口径。

三个容易写错的细节：

1. \(q_d=0\) 归到**负**号那一支；
2. 打分与 topk **都跳过最后一个 page**——它是刚写入的那页，要等下次 decode 才参与，但**永远跟着加载**；
3. 预算先被现有页数截断，再减去「恒载的最后一页」，这才是真正参与竞标的名额。

## 任务

实现 `solve(q, k, page_size=16, page_budget=1)`：

- `q`：`(Lq, H, D)` float32/float64，NHD 布局，**只支持 decode 单 query，`Lq` 必须为 1**；
- `k`：`(Lkv, H, D)` 同 dtype，`Lkv >= 1`；
- `page_size`、`page_budget`：正整数。

页数 \(P=\lceil \mathrm{Lkv}/\texttt{page\_size}\rceil\)，须 \(P\ge 2\)。页内不足的位置按**输入 dtype** 的 `finfo.min` 补齐。内部一律 float64，返回按输入 dtype。

返回 4 键字典：

- `sign`：`(H, D)`，`q_d > 0` 取 `+1`，否则（含 0）取 `-1`；
- `page_max`：`(H, P, D)`，`k * sign` 在页内的 amax；
- `scores`：`(H, P-1)`，`(q * sign) @ page_maxᵀ`，并**去掉最后一页那一项**；
- `selected`：`(H, K+1)` int64、每行升序，其中 `K = min(page_budget, P) - 1` 个页号按分数降序挑出（同分取更小的页号），末尾恒接 `P-1`。

非法输入抛 `ValueError`：非三维、`Lq != 1`、`Lkv < 1`、head 数或 head_dim 不一致、两者 dtype 不同或不是 float32/float64、`page_size`/`page_budget` 不是正整数、`P < 2`。

## 示例

```
q = [[1.0, -2.0]]              # Lq=1, H=1, D=2
k = [[1.0, 3.0],
     [2.0, 1.0],
     [-1.0, -0.5],
     [0.5, -0.2],
     [9.0, 0.5],
     [9.0, -1.0]]              # Lkv=6
page_size   = 2                # P = 3
page_budget = 2                # K = min(2, 3) - 1 = 1

sign     = [[[1, -1]]]         # q_1 = -2 < 0 → -1
max_key  = k * sign = [[[1, -3], [2, -1], [-1, 0.5], [0.5, 0.2], [9, -0.5], [9, 1]]]
page_max = [[[2, -1], [0.5, 0.5], [9, 1]]]      # 页 0 / 1 / 2 的逐通道 amax
scores   = (q*sign) @ page_max^T
         = [[[0.0, 1.5, 11.0]]][:, :, :2] = [[[0.0, 1.5]]]
selected = [[[1, 2]]]          # 页 1 在竞标中胜出；页 2 恒载
```

页 2 的上界 11.0 比胜出的页 1（1.5）高得多，但它的那一列根本没进 `scores`——最后一页恒载，让它参与竞标等于白占一个名额。

## 约束

- **零值归负**：符号是 `(q > 0) + (~(q > 0)) * (-1)`，`q_d = 0` 时取 `-1`。换成 `torch.sign` 会得到 0，整页上界直接塌成 0，也不会选到 `min` 那一端。
- **最后一页恒载但不可选**：`scores` 的列数比页数少 1；`selected` 的最后一列恒为 `P-1`，且每行升序。
- **预算先截断页数再减 1**：`K = min(page_budget, P) - 1`。`page_budget >= P` 时 `K = P-1`，等价于全加载；`page_budget = 1` 时 `K = 0`，只加载最后一页。
- **打包哨兵不要换成 0**：不足一页的位置按 `finfo(dtype).min` 补齐。amax 只被真实值影响，但如果把哨兵写成 0，某通道整页为负时 `page_max` 会被抬到 0，上界不再是上界。
- **打分是上界不是注意力**：没有 softmax、没有除 `√d`，就是原始的符号加权点积，量级随 `D` 与 key 幅值线性增长。
- **同分取更小的页号**：上游 `torch.topk` 不保证同分顺序，本题把它钉成确定性约定。
