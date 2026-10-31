# 上下文并行：zigzag token 切分与因果负载

## 任务

在单进程里模拟上下文并行（context parallel, CP）的 token 切分：把一条序列按 $2C$ 对齐补到整块，切成 $2C$ 块后按两种布局分给 $C$ 个 rank，并算出每个 rank 的因果注意力负载。

- **contiguous**：rank $r$ 拿第 $2r$、$2r+1$ 块（等价于一段连续长块）。
- **zigzag**：rank $r$ 拿第 $r$ 块与第 $2C-1-r$ 块（前一块接后一块）。

因果掩码下，越靠后的 token 要看的 key 越多，所以 contiguous 切分天然不均衡；zigzag 把后半段对称地分给靠前的 rank，在无 padding 时能把每个 rank 的 query-key 对数**精确拉平**。本题只做布局与计数，不算注意力数值。

## 输入

- `seq_len`：Python `int`，`1 <= seq_len <= 65536`。
- `cp_degree`：Python `int`，`1 <= cp_degree <= 256`。
- `layout`：`"contiguous"` 或 `"zigzag"`，区分大小写。

`bool` 冒充 `int`、浮点、越界、`layout` 拼错或不是字符串——都抛出 `ValueError`。

## 精确规则

记 $C=\texttt{cp\_degree}$，对齐粒度 $A=2C$，补齐长度

$$
P=\left\lceil \frac{N}{A}\right\rceil A,
$$

每块长度 $K=P/(2C)$，每个 rank 的局部长度 $2K$。全局补齐序列切成 $2C$ 块，第 $c$ 块覆盖原序列下标 $[cK,\,(c+1)K)$；下标 $\ge N$ 的位置是 padding。

**归属。**

- `contiguous`：rank $r$ 的第 1 块是 $2r$，第 2 块是 $2r+1$。
- `zigzag`：rank $r$ 的第 1 块是 $r$，第 2 块是 $2C-1-r$。

局部序列先放第 1 块、再放第 2 块。局部位置写**原序列下标**（0 起），padding 写 `-1`。

**因果负载。**

对 rank 持有的每个真实 token（原序列下标 $i$），因果注意力下它看见原序列位置 $\le i$ 的全部 key，共 $i+1$ 个。于是

$$
\text{causal\_pairs}[r]=\sum_{i\in \text{rank }r,\ i<N}(i+1).
$$

padding 不参与计数。

**失衡比。**

$$
\text{imbalance}=\frac{\max_r \text{causal\_pairs}[r]}{\min_r \text{causal\_pairs}[r]}.
$$

若 $\min=0$：所有 rank 都为 0 时给 `1.0`，否则给 `float("inf")`（$C>N$ 时会有 rank 一个真实 token 都拿不到）。

## 返回

返回四元 `tuple`：

| 项 | 形状 / 类型 | 含义 |
| --- | --- | --- |
| `local_indices` | `list[list[int]]`，$C\times 2K$ | 每个 rank 的局部 token 下标，padding 为 `-1` |
| `local_mask` | `list[list[bool]]`，$C\times 2K$ | `True` 表示真实 token |
| `causal_pairs` | `list[int]`，长度 $C$ | 每个 rank 的因果 query-key 对数 |
| `imbalance` | `float` | `max/min`，零最小值的约定见上 |

## 示例

```text
N = 8, C = 2, layout = "zigzag"
  P = 8, K = 2, 局部长度 4
  rank 0：块 0 + 块 3 = [0, 1, 6, 7]   负载 1+2+7+8 = 18
  rank 1：块 1 + 块 2 = [2, 3, 4, 5]   负载 3+4+5+6 = 18
  imbalance = 1.0

N = 8, C = 2, layout = "contiguous"
  rank 0：[0, 1, 2, 3] -> 10
  rank 1：[4, 5, 6, 7] -> 26
  imbalance = 2.6

N = 6, C = 2, layout = "zigzag"
  P = 8, K = 2，块 3 全是 padding
  rank 0：[0, 1, -1, -1] -> 3
  rank 1：[2, 3, 4, 5]   -> 18
  imbalance = 6.0
```

第一个例子说明 zigzag 的平衡性：rank 0 用「最前一块 + 最后一块」，rank 1 用中间两块，前块的负载差恰好被后块补回。第三个例子说明 padding 会打破平衡——尾部补出来的空位落在 zigzag 的 back 块上，只压低了持有它的 rank。

## 与相邻题目的边界

- `distributed-ring-attention` 算的是环上 KV 轮转后的在线 softmax 与整块跳过；本题不碰注意力数值，只产出静态 token 布局与工作量计数。
- `distributed-sequence-parallel` 是激活显存/通信字节的闭合账；本题不涉及字节。
- `distributed-tp-split-rearrange` 是 Ulysses 式的 head 维切分与 all-to-all；本题是 ring CP 的 token 维切分，head 维在各 rank 上复制。
- `quant-duquant-zigzag-permute` 是量化里的锯齿排列，与 CP 分块无关。
