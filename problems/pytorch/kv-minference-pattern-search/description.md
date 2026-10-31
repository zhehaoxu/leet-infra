# KV Cache：MInference 稀疏 attention pattern 搜索

## 题目描述

长上下文 prefill 的全量注意力是 \(O(N^2)\) 的，但真实模型的注意力高度稀疏且**形状可预测**：一小撮人人都看的"垂直"列（attention sink 与几个关键 token），加一条紧贴对角的"斜线"带。MInference（2407.02490）的 `search_pattern` 先做一次便宜的**模式搜索**：给一批候选 `(kind, 尺寸)`，估出「这个掩码留下了多少注意力质量」，每个 head 各挑分数最高的那个，之后才用选中的 pattern 去跑真正的稀疏 attention。

打分不需要任何 kernel。设真实因果注意力为 \(P\)（每个 head 一份 \(N\times N\) 的行随机下三角），候选掩码为 0/1 的 \(M\)，分数就是

\[
\text{score}=\frac{1}{N}\sum_{i=0}^{N-1}\sum_{j=0}^{N-1} P_{ij}\,M_{ij}
\]

即「每行被保住的注意力质量」对**全部** \(N\) 行取平均，值域 \([0,1]\)。三种 `kind` 的 \(M\) 分别是：

- `stream_llm(a, b)`：前 \(a\) 列 \(\cup\) 最近 \(b+1\) 条对角线（含主对角线），再与因果下三角取交；
- `vertical_and_slash(a, b)`：只看**最后 64 个 query** 的注意力，求每列质量 `vertical` 与每条对角线质量 `slash`；前 30 列与最近 30 条对角线被置 \(+\infty\) 强制保留，其余按质量取 top-\(a\) 列与 top-\(b\) 条，两者并起来；
- `block_sparse(a)`：\(q\)、\(k\) 补零到 \(\lceil N/32\rceil\cdot 32\) 行，每 32 行取均值得到块向量（**补进去的 0 也进分母**），块级点积**不除** \(\sqrt D\)、加块因果掩码，每行丢掉最低分的前 \(\texttt{block\_num}-\lfloor \texttt{block\_num}/\texttt{a}\rfloor\) 个块，块网格摊回 token 网格后再与因果取交。

三个容易写错的细节：

1. \(P\) 只算一次、三个候选共用；分数**不重归一化**，某行被整行掩掉就只贡献 0；
2. `slash` 选中的对角线是**并**进 vertical 结果的，不受该列 vertical 落选影响；
3. `pattern` 是**后处理过**的：`block_sparse` 胜出时改写成 `(1000, 6096)`，`stream_llm` 与 `vertical_and_slash` 都原样返回入参 `(a, b)`。

## 任务

实现 `solve(q, k, candidates)`：

- `q`、`k`：`(H, N, D)`，**形状相同、dtype 相同**，float32 或 float64，`N >= 64`、`D >= 1`；
- `candidates`：非空 list/tuple，每项是 `(kind, a, b)`；`kind` 取上面三种之一，`a`、`b` 是 int（`bool` 不算）。

\(P\) 在题内自算：`softmax(q @ kᵀ / √D)` 配因果掩码。内部一律 float64，`scores` 按输入 dtype 返回。

返回 4 键字典：

- `scores`：`(H, C)`，输入 dtype，`(h, c)` 项 = head `h` 在候选 `c` 下保住的注意力质量；
- `best`：`(H,)` int64，逐 head 的 `argmax`，**同分取更小的候选下标**；
- `pattern`：`(H, 2)` int64，胜出候选后处理后的 `(vertical_size, slash_size)`，`block_sparse` 胜出时为 `(1000, 6096)`；
- `mask`：`(H, N, N)` bool，各 head **胜出那一个**候选的掩码。

非法输入抛 `ValueError`：非三维、`q`/`k` 形状不同、两者 dtype 不同或不是 float32/float64、`N < 64`、`D < 1`、`candidates` 不是非空 list/tuple、某项不是长度为 3 的 list/tuple、未知 `kind`、`a`/`b` 不是 int、`block_sparse` 的 `a < 1`、其余 kind 的 `a` 或 `b` 落在 `[0, N]` 之外。

## 示例

```
q = k = zeros((1, 64, 4), float64)        # H=1, N=64, D=4
candidates = [("stream_llm", 30, 8),
              ("vertical_and_slash", 30, 30),
              ("block_sparse", 8, 1)]

attn[i, j] = 1 / (i + 1)  for j <= i      # q=0，注意力在因果前缀上均匀
scores  = [[0.9081807300938489, 0.9975192100819044, 0.0]]
best    = [1]
pattern = [[30, 30]]
mask[0, 63] = 列 0..29 + 列 34..63        # 30 个 vertical 列 + 最近 30 条对角线
```

`block_sparse` 只有 \(\texttt{block\_num}=2\) 个块，`topk_ratio=8` 时 `2 // 8 = 0`，一个块都留不下 → 掩码全空、分数 0。同样保住 30 个 vertical 列，`slash` 从 8 条放宽到 30 条就把质量从 0.9082 抬到 0.9975；第 63 行被丢的 30..33 正是"vertical 落选又没被任何一条斜线带覆盖"的列。

## 约束

- **vertical/slash 只看最后 64 个 query**：`N > 64` 时前面的行完全不参与统计，`N = 64` 时才等于全体——这就是 `N >= 64` 的由来。
- **强制保留是"置 \(+\infty\) 再排序"，不是"跳过"**：`a < 30` 或 `b < 30` 时那 30 个并列 \(+\infty\) 的条目本身也要参与排序，同分取更小下标 → `vertical` 从第 0 列开始被吃掉、`slash` 从最老的 \(\delta=-29\) 那条开始被取。
- **块均值分母恒为 32**：最后一个不满块补进去的 0 也进 `mean`，块向量被压小、块分数随之下降；按真实条数重归一化会选到别的块。
- **名额是"丢掉最低的"**：`vertical` 丢 `N - a` 列、`block` 丢 `block_num - block_num // a` 块；`block_num < topk_ratio` 时一个都留不下。
- **两处缩放口径不同**：\(P\) 除 \(\sqrt D\)，`block_sparse` 的块级点积不除。
- **`block_sparse` 的 `b` 完全不参与计算**：题面仍要求它是 int，但**不做 `[0, N]` 范围检查**。
- **`mask` 逐 head 取胜出者**：不是三个候选的并；`stream_llm` 的掩码对所有 head 相同，另两种逐 head 不同。
