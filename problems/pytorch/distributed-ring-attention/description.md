# Ring Attention：KV 环转、在线 softmax 与因果整块跳过

## 题目描述

序列/上下文并行把一条长序列切成 C 段分别放在 C 张卡上，每张卡只持有自己那段的 Q、K、V。
但注意力的每个 query 都要看**全序列**的 key，于是 Ring Attention（Liu 等, 2023）让 KV 在环上
转起来：第 $s$ 步，rank $r$ 处理上游传来的那块 KV，算完再把自己手里的那块传给下游。转满 $C$
步，每个 rank 就在**从不物化整行 $L \times L$ 分数矩阵**的前提下，看完了所有 key。

转起来以后有两个工程后果，本题就要把它们逐格算清：

- **在线 softmax**：每一步只拿到一块 key，运行最大值 $m$ 与分母 $l$ 必须能在新块到来时
  **就地修正**，而不是攒完所有 logit 再算一遍。
- **因果整块跳过**：因果掩码下，rank $r$ 的 query 只看得见位置 $\le$ 自己的 key。环上晚到的
  那些块来自比自己**更靠后**的序列段，整块都不可见——可以整块跳过，一块 exp 都不做。
  但块**照样被收下、照样被转发**，省的是算力，不是带宽。

## 模型（务必按此定义）

### 布局与步

- `q_shards`、`k_shards`、`v_shards` 各有 $C \ge 1$ 个分片，每片 $S \ge 1$ 行、每行 $D \ge 1$ 个数。
- 全局序列是分片**按 rank 下标拼接**：$N = C\cdot S$，rank $r$ 的第 $i$ 行就是全局第 $rS+i$ 行。
- 第 $s$ 步（$0 \le s < C$）rank $r$ 收到 KV 分片
  $$ \text{src} = (r - s) \bmod C $$
  于是每个 rank 在 $C$ 步里把每个分片恰好见一次，第 0 步见到的就是自己那块：
  rank $r$ 的源序列是 $r,\ r-1,\ \dots,\ 0,\ C-1,\ \dots,\ r+1$。

### 整块跳过

`causal` 为真且 $\text{src} > r$ 时，该块的 key 全局位置是 $[\text{src}\cdot S,\ \text{src}\cdot S + S - 1]$，
全部 $> rS + S - 1$（本 rank 最后一行的位置），**整块不可见**：直接跳过，不做 exp、不改状态。

判据写成位置比较就是 $\text{src}\cdot S > rS + S - 1$；写成下标就是 $\text{src} > r$。
`causal` 为假时一块都不跳。

### 在线 softmax

logit 为 $a = \text{scale}\cdot(q_{\text{row}} \cdot k_{\text{row}})$；因果掩码只保留全局 key 位置
$\le rS+i$ 的 key（所以自己的块是三角，更早的块全保留）。每行维护运行最大值 $m$、分母 $l$、
分子 $\text{acc}$（$D$ 维），处理一个块时：

$$ \begin{aligned}
b &= \max_{\text{保留的} j} a_j & &\text{(掩码后至少还剩一个 key)}\\
m' &= \max(m, b)\\
\text{corr} &= \exp(m - m') & &(m = -\infty \text{ 时自然得 } 0.0)\\
l &\leftarrow l\cdot\text{corr} + \sum_j \exp(a_j - m')\\
\text{acc}_d &\leftarrow \text{acc}_d\cdot\text{corr} + \sum_j \exp(a_j - m')\,v_{jd}\\
m &\leftarrow m'
\end{aligned} $$

$C$ 步走完，$\text{out}[r][i][d] = \text{acc}_d / l$。

## 规则

实现 `solve(q_shards, k_shards, v_shards, causal, scale, dtype_bytes)`，返回五元组
`(out, traces, sources, skipped, comm_bytes)`：

- `out`：$C$ 个分片、每片 $S$ 行 $D$ 列的 float，即 $\text{acc}/l$。
- `traces`：`traces[r][s][i]` 是第 $s$ 步**结束后**第 $i$ 行的 `(m, l)`。被整块跳过的步原样
  **照抄上一步**（第 0 步永远不会被跳过，因为它必然是自己的块）。
- `sources`：`sources[r][s] = src`，int。
- `skipped`：bool，标记该步是否整块被因果掩码吃掉。
- `comm_bytes`：`comm_bytes[r][s] = 2 * S * D * dtype_bytes`（当 $s < C-1$），否则 `0`。
  2 指「收一块 + 发一块」；最后一步之后环上不再有下一跳，所以每个 rank 只付 $C-1$ 次。
  **被跳过的块照样计入**。

非法入参一律抛 `ValueError`：分片列表为空、三个参数的 rank 数不一致、分片行数或行宽不一致、
元素不是有限的 int/float（`bool`、`None`、复数、字符串、`inf`、`nan` 都不算）、`causal` 不是
`bool`（`0/1` 不算）、`scale` 不是严格正、`dtype_bytes` 不是正整数。

## 示例

**例一**：$C=2$、$S=2$、$D=1$、`causal=True`、`scale=1.0`、`dtype_bytes=4`。
q 全零，k 与 v 都是 `[[[1.0], [2.0]], [[3.0], [4.0]]]`（全局 key 依次是 1、2、3、4）。
logit 全为 0，所以输出就是「可见 key 的均值」：第 0 行只看得见 1，第 3 行全看得见。

```
out     = [[[1.0], [1.5]], [[2.0], [2.5]]]
traces  = [[[(0.0, 1.0), (0.0, 2.0)], [(0.0, 1.0), (0.0, 2.0)]],
           [[(0.0, 1.0), (0.0, 2.0)], [(0.0, 3.0), (0.0, 4.0)]]]
sources = [[0, 1], [1, 0]]
skipped = [[False, True], [False, False]]
comm    = [[16, 0], [16, 0]]
```

rank 0 的第 1 步拿到 src=1，整块在它所有 query 之后，于是跳过：`traces[0][1]` 与
`traces[0][0]` 一模一样，但 `comm[0][0] = 2*2*1*4 = 16` 照付。

**例二**：$C=2$、$S=1$、$D=1$、`causal=True`、`scale=1.0`、`dtype_bytes=4`，
q 全为 1，k 与 v 都是 `[[[1.0]], [[2.0]]]`。rank 1 的两步 logit 分别是 2 与 1：

```
out     = [[[1.0]], [[1.7310585786300049]]]
traces  = [[[(1.0, 1.0)], [(1.0, 1.0)]],
           [[(2.0, 1.0)], [(2.0, 1.3678794411714423)]]]
sources = [[0, 1], [1, 0]]
skipped = [[False, True], [False, False]]
comm    = [[8, 0], [8, 0]]
```

rank 1：先 $m=2$、$l=1$；第二步 $\text{corr} = \exp(2-2) = 1$，
$l = 1 + \exp(1-2) = 1.3678794411714423$，$\text{acc} = 2 + \exp(-1)\cdot 1$，
$\text{out} = 2.3678794411714423 / 1.3678794411714423 = 1.7310585786300049$。

**例三**：把例一的 `causal` 改成 `False`。一块都不跳，每个 query 都看得见全部 4 个 key：

```
out     = [[[2.5], [2.5]], [[2.5], [2.5]]]
traces  = [[[(0.0, 2.0), (0.0, 2.0)], [(0.0, 4.0), (0.0, 4.0)]],
           [[(0.0, 2.0), (0.0, 2.0)], [(0.0, 4.0), (0.0, 4.0)]]]
sources = [[0, 1], [1, 0]]
skipped = [[False, False], [False, False]]
comm    = [[16, 0], [16, 0]]
```

**例四**：$C=3$、$S=1$、$D=1$、`causal=True`、`scale=1.0`、`dtype_bytes=2`，
q 为三个 0，k 与 v 都是 `[[[1.0]], [[2.0]], [[4.0]]]`。环要倒着转两格才回到自己：

```
out     = [[[1.0]], [[1.5]], [[2.3333333333333335]]]
traces  = [[[(0.0, 1.0)], [(0.0, 1.0)], [(0.0, 1.0)]],
           [[(0.0, 1.0)], [(0.0, 2.0)], [(0.0, 2.0)]],
           [[(0.0, 1.0)], [(0.0, 2.0)], [(0.0, 3.0)]]]
sources = [[0, 2, 1], [1, 0, 2], [2, 1, 0]]
skipped = [[False, True, True], [False, False, True], [False, False, False]]
comm    = [[4, 4, 0], [4, 4, 0], [4, 4, 0]]
```

rank 0 的两步都被跳过（src 2、1 都在它后面），于是它的 `traces` 三步全是 `(0.0, 1.0)`；
rank 2 一块都没被跳过。$C=1$ 时退化为「只看自己的块」：`sources = [[0]]`、
`skipped = [[False]]`、`comm = [[0]]`。

## 约束

- $1 \le C \le 32$、$1 \le S \le 64$、$1 \le D \le 64$；`dtype_bytes` 取 1、2、4、8 之一
- 元素与 `scale` 保证所有 logit 都是有限值，不必处理 `inf` / `nan`
- `out` 与 `traces` 的 float 按相对 $10^{-12}$ 判等；`sources`、`skipped`、`comm_bytes` 精确判等
- 复杂度上限：单次调用 $O(C^2 S^2 D)$，最大规模约 $10^7$ 次乘加
