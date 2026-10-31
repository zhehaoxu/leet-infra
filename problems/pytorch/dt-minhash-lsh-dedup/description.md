# MinHash 签名与 LSH 分带近重复去重

## 题目描述

语料去重在亿级文档上不可能两两比集合：十万篇文档就是五十亿对，每对还要算一次交集。工业管线的做法是把「谁跟谁像」拆成两步——先用 **MinHash** 把每篇文档压成一个定长签名（同一置换下，两篇文档签名格相等的概率恰等于它们的 Jaccard 相似度），再用 **LSH 分带**把「找相似对」变成「找桶邻居」：把签名切成 `bands` 段、每段 `rows` 格，只有某一段完全相同的文档才进同一个桶，桶内两两成候选对；候选对再拿原始 shingle 集合做一次**精确复核**。段数与每段行数的乘积决定召回率与候选量——`rows` 越大越严格、候选越少，`bands` 越大越宽松。这正是 GPT/LLaMA 类语料管线（CC-Net、Dolma、RedPajama 的 minhash-dedupe）里那套算法的骨架。

本题把这套骨架完整走一遍：签名、分带、候选对、精确复核、连通分量聚簇、保留清单，五个中间结果都要交出来，任何一步做错都会在下游暴露。为了判题确定，**置换系数由入参显式给出**，没有随机数、没有分词器、没有哈希库——全程纯 Python 整数运算。

## 输入

- `docs`：`list[list[int]]`，每篇文档是一串非负整数 token id（已切好）。篇数与每篇长度都可以为 0。
- `n`：正整数，shingle 长度（连续窗口宽度）。
- `perms`：非空 `list`，长度记 `num_perm`，每项是 `(a, b)` 两个数组成的二元组（`tuple` 或 `list` 均可），即一个置换的系数。
- `bands`、`rows`：正整数，分带参数，必须满足 `bands * rows == num_perm`。
- `threshold`：浮点数，`0.0 <= threshold <= 1.0`，复核阶段的相似度阈值。

**禁止**调用任何现成的 MinHash / LSH / 去重 / 相似度库；签名、桶、并查集都要手写。

## 精确规则

记 `BASE = 31`、`MOD = 2305843009213693951`（即 \(2^{61}-1\)），全程整数运算按 `MOD` 取模。

1. **shingle 集**。文档 `d` 取所有连续 `n` 窗口（重叠，共 `max(0, len(d) - n + 1)` 个）。每个窗口 `w` 折成一个整数：

   `sid = 0`；对 `w` 中每个 `t` 依次做 `sid = (sid * 31 + t) % MOD`。

   \(S(d)\) 是这些 sid 构成的**集合**——同一个 sid 出现多次只算一次。`len(d) < n` 时 \(S(d)=\varnothing\)。

2. **签名**。置换 `p = (a, b)` 定义 \(h_p(x) = (a \cdot x + b) \bmod \text{MOD}\)，值域是 \([0, \text{MOD})\)。

   `sig[i][p] = min(h_p(x) for x in S(docs[i]))`；若 \(S(d_i)=\varnothing\)，该文档**每一格**都取 `MOD`（min 的单位元，刻意落在合法值域之外，便于识别空集文档）。签名按输入文档顺序排列，形状 \((N, \text{num\_perm})\)。

3. **分带**。第 `k` 段（`0 <= k < bands`）取签名的**连续**切片 `sig[i][k*rows : (k+1)*rows]`（不是跨步取样）。段值完全相同的文档构成一个桶，桶成员两两配对得到该段的候选对。总候选对是各段的**并集去重**：一对文档即使在多个段都撞上也只计一次。每对写成 `(p, q)` 且 `p < q`，整体按字典序升序。

4. **精确复核**。对每个候选对算

   \[ J(p,q) = \frac{|S(d_p)\cap S(d_q)|}{|S(d_p)\cup S(d_q)|}, \qquad \text{并集为空时记 } 0.0 \]

   判定用 Python 浮点除法：`J(p, q) >= threshold` 才算重复（**取等号算重复**）。通过的进 `duplicates`，同样 `(p, q)`、去重、字典序升序。`duplicates` 必须是 `candidates` 的子集——没有被分带提出的对，即使真重复也不许补上。

5. **聚簇与保留**。以 `duplicates` 为无向边取**连通分量**（传递闭包：`0~1` 且 `1~2` 则 `{0,1,2}` 同簇；孤立文档自成一簇）。`labels[i]` 写 \(i\) 所在分量的**最小下标**，`kept` 写各分量代表（即最小下标）的升序列表，等价于 `sorted(set(labels))`。被去掉的文档就是不在 `kept` 里的那些。

## 返回

五元组 `(sig, candidates, duplicates, labels, kept)`，全部是 Python 内置类型：

- `sig`：`list[list[int]]`，形状 \((N, \text{num\_perm})\)，空集文档的格值为 `MOD`。
- `candidates`：`list[tuple[int, int]]`，去重、`p < q`、字典序升序。
- `duplicates`：`list[tuple[int, int]]`，`candidates` 的子集，同种序约束。
- `labels`：`list[int]`，长度 \(N\)，值是分量最小下标。
- `kept`：`list[int]`，升序。

非法入参一律抛 `ValueError`：`docs` 不是 list、内层不是 list、元素不是 `int`（`bool` 不算 `int`）或为负；`n` 不是 `int` 或 < 1；`perms` 为空、不是 list、某项长度不为 2、系数不是 `int`、`a` 不在 \([1, \text{MOD})\)、`b` 不在 \([0, \text{MOD})\)；`bands` 或 `rows` 不是 `int` 或 < 1；`bands * rows != num_perm`；`threshold` 不是 `int`/`float`（`bool` 不算）或不在 \([0, 1]\)。\(N=0\) 是合法输入，返回空的 `sig`/`candidates`/`duplicates`/`labels`/`kept`。不得修改 `docs` 或 `perms`。

## 示例

```
docs = [[1, 2, 3], [1, 2, 3, 2, 3], [4, 5], [7]]
n = 2, perms = [(1, 0), (3, 7), (5, 1), (2, 3)], bands = 2, rows = 2, threshold = 0.5

# shingle sets, sid = Horner with base 31 modulo 2**61-1
S0 = {33, 65}          # (1,2)=33   (2,3)=65
S1 = {33, 65, 95}      # windows (1,2) (2,3) (3,2) (2,3); the repeated (2,3) counts once
S2 = {129}             # (4,5)=129
S3 = set()             # len 1 < n

sig = [[33, 106, 166, 69],
       [33, 106, 166, 69],
       [129, 394, 646, 261],
       [M, M, M, M]]            M = 2305843009213693951, the empty-set sentinel

# rows = 2, so band0 = sig[0:2] and band1 = sig[2:4] (contiguous, not strided)
band0: (33, 106) -> {0, 1}    (129, 394) -> {2}    (M, M) -> {3}
band1: (166, 69) -> {0, 1}    (646, 261) -> {2}    (M, M) -> {3}

candidates = [(0, 1)]          # {0, 1} collides in both bands but counts once
J(0, 1) = |{33, 65}| / |{33, 65, 95}| = 2/3 = 0.6666666666666666 >= 0.5
duplicates = [(0, 1)]
labels = [0, 0, 2, 3]
kept   = [0, 2, 3]             # doc 1 is dropped, doc 3 survives alone
```

分带只负责提名，判不判重复由精确 Jaccard 说了算。`rows = 1` 时提名会明显变松：

```
docs = [[1, 2, 3], [1, 4, 5], [2, 3, 6], [1, 2, 3], [], []]
n = 1, perms = [(1, 0), (2, 0)], bands = 2, rows = 1, threshold = 0.7

S0 = {1, 2, 3}   S1 = {1, 4, 5}   S2 = {2, 3, 6}   S3 = {1, 2, 3}   S4 = S5 = set()
sig = [[1, 2], [1, 2], [2, 4], [1, 2], [M, M], [M, M]]

band0 (sig[0]): 1 -> {0, 1, 3}    2 -> {2}    M -> {4, 5}
band1 (sig[1]): 2 -> {0, 1, 3}    4 -> {2}    M -> {4, 5}

candidates = [(0, 1), (0, 3), (1, 3), (4, 5)]
J(0, 1) = 1/5 = 0.2  <  0.7   rejected
J(0, 3) = 3/3 = 1.0  >= 0.7   duplicate
J(1, 3) = 1/5 = 0.2  <  0.7   rejected
J(4, 5) = 0.0 (union is empty)  <  0.7   rejected
duplicates = [(0, 3)]
labels = [0, 1, 2, 0, 4, 5]
kept   = [0, 1, 2, 4, 5]       # only doc 3 is dropped
```
