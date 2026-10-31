# 上下文并行：变长打包序列的 zigzag 切分计划

## 任务

真实训练里一个 batch 常把多条变长序列**打包**成一条 token 流（THD 布局，`cu_seqlens` 记录段边界）。上下文并行要把这条流分给 $C$ 个 rank，同时满足两个约束：每段内部按 zigzag 前后配对、每个 rank 的局部长度相等。

本题产出这份布局计划：每段先补齐到 $2C$ 的整数倍，再切块配对，最后给出**每个 rank 的局部槽位里应该放哪个原始 token 下标**（padding 为 `-1`），以及补齐后的段边界。

## 输入

- `cu_seqlens`：长度 $B+1\ge 2$ 的 `list`/`tuple`，元素是非 `bool` 的 Python `int`；`cu_seqlens[0] == 0`，严格递增，末项 $\le 65536$（因此每段长度 $\ge 1$）。
- `cp_degree`：Python `int`，`1 <= cp_degree <= 256`。

起点非 0、非严格递增、负数、类型不符、越界、`cp_degree` 非法——都抛出 `ValueError`。

## 精确规则

记 $C=\texttt{cp\_degree}$，第 $b$ 段长度 $n_b=\texttt{cu\_seqlens}[b+1]-\texttt{cu\_seqlens}[b]$，对齐粒度 $A=2C$。

**每段补齐。**

$$
p_b=\left\lceil\frac{n_b}{A}\right\rceil A,\qquad P=\sum_b p_b,\qquad L=\frac{P}{C}.
$$

$P$ 一定是 $2C$ 的倍数，所以 $L$ 是整数。补齐后的段边界是前缀和 $q_b=\sum_{j<b}p_j$，即返回的 `cu_seqlens_padded`（首项 0）。

**段内切块与配对。**

段 $b$ 切成 $2C$ 个等长块，块长 $k_b=p_b/(2C)$，第 $c$ 块覆盖段内偏移 $[c k_b,\,(c+1)k_b)$。rank $r$ 拿第 $r$ 块与第 $2C-1-r$ 块。

**局部位置。**

段 $b$ 的局部基址是 $q_b/C$（因为每段在每个 rank 上占 $2k_b=p_b/C$ 个槽）。前块占

$$
[q_b/C,\ q_b/C+k_b),
$$

后块紧接着占

$$
[q_b/C+k_b,\ q_b/C+2k_b).
$$

槽里写该 token 在整个打包流里的**原始下标**（跨段连续，从 0 起）；段尾补齐的位置写 `-1`。

**补齐账。**

- `pad_total` $=P-N$，$N=\texttt{cu\_seqlens}[-1]$。
- `pad_between_seqs`：只要有任何一段 $p_b\ne n_b$ 就为 `True`。padding 在各段尾部，所以可能夹在两条序列之间，这也是名字的由来。

## 返回

返回四元 `tuple`：

| 项 | 形状 / 类型 | 含义 |
| --- | --- | --- |
| `cu_seqlens_padded` | `list[int]`，长度 $B+1$ | 补齐后的段边界，首项 0 |
| `rank_order` | `list[list[int]]`，$C\times L$ | 每 rank 每个槽对应的原 token 下标，padding 为 `-1` |
| `pad_total` | `int` | $P-N$ |
| `pad_between_seqs` | `bool` | 是否有段被补齐 |

## 示例

```text
cu_seqlens = [0, 3, 5], C = 2
  n = [3, 2] -> p = [4, 4], P = 8, L = 4, 块长 = 1
  段 0 基址 0：rank0 = 块0 + 块3 = [0, -1]；rank1 = 块1 + 块2 = [1, 2]
  段 1 基址 2：rank0 = 块0 + 块3 = [3, -1]；rank1 = 块1 + 块2 = [4, -1]
  cu_seqlens_padded = [0, 4, 8]
  rank_order = [[0, -1, 3, -1],
                [1,  2, 4, -1]]
  pad_total = 3, pad_between_seqs = True

cu_seqlens = [0, 4], C = 2
  n = [4] -> p = [4], L = 2
  rank_order = [[0, 3],
                [1, 2]]
  cu_seqlens_padded = [0, 4], pad_total = 0, pad_between_seqs = False
```

第一个例子里段 0 补了 1 个位置、段 1 补了 2 个，padding 分别落在两段的尾部；rank 0 的局部序列恰好把两段的首块与末块拼在一起。第二个例子完全整除，退化成单序列 zigzag 切分。

## 与相邻题目的边界

- `distributed-cp-zigzag-partition` 是**单条定长序列**的 zigzag 切分与因果负载；本题是**多段打包**的布局计划，核心难点在每段各自补齐后的跨段槽位映射。
- `distributed-ring-attention` 做环上 KV 轮转与在线 softmax；本题只产出静态下标计划。
- `distributed-sequence-parallel` 是激活/通信字节的闭合账；本题不涉及字节。
- `distributed-tp-split-rearrange` 是 Ulysses 的 head 维切分；本题仍是 token 维。
