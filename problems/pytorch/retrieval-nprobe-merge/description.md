# 检索：nprobe 多探测与跨 cell 归并

## 题目描述

IVF 查询的性能与召回都由一个旋钮控制：**nprobe**，即一次查询访问多少个倒排链（Faiss `IndexIVF` 的 `nprobe` 参数；Jégou, Douze, Schmid, *Product Quantization for Nearest Neighbor Search*, IEEE TPAMI 2011, arXiv:1009.5545 中的 coarse quantizer + inverted list 结构）。nprobe 越大，扫到的候选越多、召回越高，但延迟也越高。本题只实现这次查询里与「多探测」有关的三步确定性逻辑：

1. 按 query 到粗中心的**平方欧氏距离**给 cell 排序，取前 nprobe 个（并列取小 cell 号）；
2. **按这个顺序**逐个探测 cell，把各链的 posting 取并集；同一个全局 id 可能在多个链里出现，只保留一次，距离取所有出现中的**最小值**；
3. 对去重后的候选按 **(距离升序, 全局 id 升序)** 排序，取 top-k。

为了能观测中间过程，除了 top-k 结果，还要输出**每步选中的 cell 序列**与**候选数统计**（扫描了多少 posting、去重后剩多少、删掉多少重复、最终返回多少）。倒排表存完整向量，距离就是 query 与该向量的平方欧氏距离；不涉及 PQ 近似。

CPU、全程确定性、不调用任何随机数。

## 规则

完整签名、参数含义与默认值只写在 starter 的 docstring 里，那是唯一判题口径。

- **输入**：查询 \((Q,D)\) 浮点张量（float32/float64，\(Q\) 可为 0）；粗中心 \((C,D)\)；倒排表用两个长度均为 \(C\) 的 list 表示，第 \(c\) 项分别是 \((n_c,D)\) 的完整向量与 \((n_c,)\) 的 int64 全局 id，\(n_c\) 可为 0。允许同一个 id 在同一个或不同 cell 中出现多次。
- **探测顺序**：对每个查询计算 \(\|q-c_j\|^2\)，按升序取前 \(P=\min(\text{nprobe},C)\) 个 cell，并列取 cell 号小者。`nprobe=0` 时 \(P=0\)，不探测；`nprobe>C` 时 \(P=C\)。探测必须按这个顺序进行，而不是按 cell 号顺序。
- **单条距离**：\(\|q-v\|^2\)，逐维平方差求和，内部用 float64 累加，输出时转回查询 dtype。
- **归并与去重**：按探测顺序收集 posting；同一全局 id 出现多次时只保留一次，距离取这些出现中的最小值。没有探测到任何 posting 时，统计四列全 0。
- **top-k**：去重后的候选按 (距离升序, 全局 id 升序) 排序，取前 \(k\) 个；不足 \(k\) 个时尾部填充。
- **输出四个张量**：
  - `ids`：\((Q,k)\) int64，真实候选之后填 \(-1\)；
  - `dists`：\((Q,k)\)，dtype 与查询相同，填充位为 \(+\infty\)；
  - `cells`：\((Q,P)\) int64，每行是实际探测的 cell 号序列；`nprobe=0` 时为 \((Q,0)\)，`nprobe>C` 时 \(P=C\)；
  - `stats`：\((Q,4)\) int64，四列依次为 `[n_postings, n_unique, n_duplicates, n_returned]`，其中 `n_postings` 是扫描的 posting 总数（含重复），`n_unique` 是不同 id 数，`n_duplicates = n_postings - n_unique`，`n_returned = min(k, n_unique)`。
- **退化**：`k=0` 时 `ids`/`dists` 形状为 \((Q,0)\)，但 `cells` 与 `stats` 仍按实际探测结果填写（`n_returned=0`）；`C=0` 时 `cells` 为 \((Q,0)\)、`stats` 全 0。
- **错误约定**：维度不符、\(D\le 0\)、dtype 不一致或非 float32/float64、倒排表长度不等于 \(C\)、vectors 形状或 dtype 不符、ids 不是 1-D int64、ids/vectors 行数不一致、id 为负、`nprobe<0`、`k<0`，一律抛 `ValueError`。

## 示例

查询 \(q=(0,0)\)，粗中心 \(c_0=(0,0)\)、\(c_1=(3,0)\)、\(c_2=(1,0)\)，到 \(q\) 的平方距离分别是 0、9、1。取 `nprobe=2`，探测顺序为 **cell 0 → cell 2**（cell 2 比 cell 1 近）。

倒排表：

| cell | posting：id / 向量 | 到 q 的平方距离 |
|---|---|---|
| 0 | 7 / (0,0)；3 / (0,0) | 0；0 |
| 1 | 8 / (3,0) | 9（未探测） |
| 2 | 3 / (2,0)；5 / (1,0) | 4；1 |

按顺序扫描得到 4 条 posting：`(7,0), (3,0), (3,4), (5,1)`。id 3 出现两次，保留最小距离 0，于是去重后 3 个候选 `(3,0), (7,0), (5,1)`；id 3 与 7 距离并列，按 id 升序排。`k=3`：

```
ids   = [[3, 7, 5]]
dists = [[0.0, 0.0, 1.0]]
cells = [[0, 2]]
stats = [[4, 3, 1, 3]]   # 扫描 4 条、去重后 3 个、删掉 1 条重复、返回 3 个
```

若取 `nprobe=1`，只探测 cell 0，候选为 id 3/7，`k=3` 的第三格填充：

```
ids   = [[3, 7, -1]]
dists = [[0.0, 0.0, inf]]
cells = [[0]]
stats = [[2, 2, 0, 2]]
```

## 约束

- 返回四元组 `(ids, dists, cells, stats)`，形状与 dtype 必须与上述口径完全一致；`cells` 与 `stats` 全程 int64。
- 判题对 `ids`、`cells`、`stats` 用 `torch.equal` 精确比对；浮点距离 float32 用 \(10^{-5}/10^{-6}\)、float64 用 \(10^{-9}/10^{-12}\) 的 `atol/rtol`。
- \(Q,C,n_c\) 均可为 0；`nprobe` 与 `k` 必须是非负整数。
- 不使用随机数；实现路径不限，但探测顺序、去重取最小距离、以及并列排序三条口径都不能走样。
