# 参考思路：四道关卡，每道都有一条容易踩空的边界

## 实现

四段串行，前一段的输出就是后一段的全部输入；把五个中间结果一起交出来，任何一段读错都会
在下游那一格暴露。

1. **温度权重**。`expo = 1.0 / T`，`e_i = w_i ** expo`，但 `sizes[i] == 0` 的源在**缩放
   之前**就强制置零。顺序很重要：若先缩放再置零，`total` 里仍会算进那个源，剩下几个源
   的概率被整体压低，配额也跟着少——而它们并没有多出来的样本可发。例二就是这一条：
   `sizes = [4, 0, 9]`、`weights = [1, 5, 4]`、`T = 2`，源 1 的 \(\sqrt 5 = 2.236\dots\)
   必须消失，剩下的 `1 : 2` 才是 `1/3 : 2/3`。
2. **归一化**。`total` 有限且 \(>0\) 才可归一化。全部权重为 0、所有源都空、或只有空源
   带权重，三种情况都落在这里抛 `ValueError`。
3. **最大余额配额**。`q = p * N`，`base = floor(q)`，`extra = N - sum(base)`，按
   `(q - base)` 降序、下标升序取前 `extra` 个加一。**`extra <= k` 由构造保证**：`base`
   与 `q` 的差是各项小数部分之和，每项 \(<1\)，所以 `extra = N - sum(base) <= k`；
   而排序结果里每个下标只出现一次，于是不会有源被加两次。平局规则用一次稳定排序实现，
   不需要手写循环。
4. **平滑排布**。`slot_0 = j * N // c` 是从理想槽起向后扫描第一个空位的起点。这里用
   **整数除法**而不是 `floor(j * N / c)`：两者在本题的量级下恒等（`j * N <= 4e10 < 2**53`，
   且 `c <= N` 使商与整数的距离至少 `1/c`），但整数除法不依赖浮点舍入。
5. **循环越采样与全局号**。第 `j` 次发射读 `j % sizes[i]`，配额超过篇数就环形复读；
   `offsets` 是 exclusive 前缀和，所以空源不占号段，`flat` 自然落在 `[0, sum(sizes))` 内。

第 4 步的「向后扫描」会不会越界？判题里没有回绕分支，因为**越界不可能发生**：
把 `N <= 300`、`1..9` 个源的随机配额向量扫了 5000 组，越界 0 次；把 `N <= 11`、`k <= 4`
的全部整数拆分（组合枚举）扫了一遍，同样 0 次。原因是同一源的理想槽间距恒为
`N/c >= 1`，前面的源只可能占掉它自己的槽位，最后一个源发射完必然还有空位。

平滑排布到底值多少？拿 `probs = [0.4, 0.3, 0.2, 0.1]`、`N = 48`（配额 `19/14/10/5`）
对比两个方案：本题的理想槽位，和「按源聚堆」的基线（同一来源的样本全排在一起）。
间隔统计按**环**算（首尾也算一段间隔），`cv` 是间隔的变异系数：

```text
# 4 sources, N = 48, quotas 19/14/10/5 (probs 0.4/0.3/0.2/0.1)
# max gap and gap coefficient of variation over the circular layout
  source 0: quota 19   smooth max gap  3  cv 0.198   | dump max gap 30  cv 2.563
  source 1: quota 14   smooth max gap  5  cv 0.286   | dump max gap 35  cv 2.554
  source 2: quota 10   smooth max gap  9  cv 0.435   | dump max gap 39  cv 2.375
  source 3: quota  5   smooth max gap 17  cv 0.565   | dump max gap 44  cv 1.792
```

聚堆基线里 40% 的主源会在 epoch 末尾连续缺席 30 个 step，理想槽位把这个数字压到 3。
小源的 `cv` 天然更大（5 个样本摊在 48 个槽上，间隔只能是 9 或 10），这不是实现误差。

## 正确性检验

十二条 `check_*`，全部与结构独立的 oracle 对拍：oracle 的 `probs` 用 `reduce` 左折叠且
先把条目转成 `float`，配额用「反复在未被补位的下标里取最大余额」（平局天然留下标小者），
排布用**并查式后继数组**跳到第一个空闲槽、理想槽走浮点 `floor`，全局号用
`sum(sizes[:src])` 现算而不是预建表。参考解是就地 `for` 循环、一次稳定排序、向后线性扫描
与整数除法、预先建 `offsets`。五条路径不共用中间结果。`_verify` 除五元组逐位比对之外
还强制：`probs` 每项是 `float` 且和与 1 的差 \(\le 10^{-9}\)；`sum(counts) == num_samples`；
`floor(q_i) <= counts_i <= floor(q_i) + 1`；`counts[i] == 0` 当 `sizes[i] == 0`；
`schedule` 里源 `i` 恰好出现 `counts[i]` 次、且 `schedule` 长度等于预算；
`0 <= doc_indices[t] < sizes[schedule[t]]`；用 `bisect` 把 `flat_indices[t]` 反解回
`(source, doc)` 必须与两列一致；`sizes` 与 `weights` 未被修改。

- `check_example`：题面四个例子的五元组逐位手算值（第三个例子同时钉死 `j % 5` 的环形复读）。
- `check_matches_oracle`：27 组 fixture，覆盖单源、空源在首位/中间/末位、权重为 0、
  预算为 0、预算小于源数、全等权、`T = 0.5/1/2/3/4/100`、`1e6` 级大源、
  `[1e6, 1e6]` 这种配额远超篇数的组合。
- `check_quota_is_largest_remainder`：`q = [0.5, 0.5], N = 1` 时四舍五入（银行家）会给出
  `[0, 0]` 而最大余额给 `[1, 0]`；`q = [1.5, 1.5], N = 3` 时逐项四舍五入会超发到
  `[2, 2]`；`q = [7/3] * 3, N = 7` 时逐项向下取整会少发到 `[2, 2, 2]`。
- `check_temperature_flattens_and_sharpens`：`T = 1` 必须逐位还原 `weights` 比例
  （`1/7, 2/7, 4/7`）；`T = 0.5` 必须是平方（`1/21, 4/21, 16/21`）且配额 `[1, 3, 10]`；
  `T = 2` 时 `[3, 5, 6]`；`T = 1e6` 时两个源的配额被推到 `[6, 7]`（几乎均分）。
- `check_cyclic_oversampling`：配额 5/4 打在 2/2 篇上，源内文档号必须是
  `[0, 1, 0, 1, 0]` 与 `[0, 1, 0, 1]`；单源 4 次发射全是 0 号；25 次发射的源 0 序列是
  13 个元素的 `0,1,2` 循环。
- `check_placement_is_by_source_order`：`counts = [2, 3]` 与 `[2, 3, 1]` 两组，
  本题的「按下标升序放置」与「按配额降序放置」结果不同，两处都比对并显式断言不等。
- `check_first_source_sits_on_ideal_slots`：对每个 fixture，取配额为正的最小下标源，
  它的槽位必须**恰好**是 `[floor(j*N/c)]`（第一个放置、无干扰，可以精确断言）。
- `check_zero_size_source_is_never_used`：`[0, 5, 0, 5]` 且空源带更大权重，仍然
  `counts = [0, 3, 0, 3]`、`set(schedule) == {1, 3}`、`flat` 跳过空号段；
  `[4, 0, 6]` 上权重 3 的空源同样拿不到配额。
- `check_flat_indices_are_global_ids`：`sizes = [3, 1, 4]`、`N = 8` 手算 `flat`。
- `check_num_samples_zero_and_tiny`：预算 0 时 `probs` 照常归一化、后三个列表为空；
  预算 2 小于源数 4 时只发 `[1, 1, 0, 0]`。
- `check_scale`：40 个源（含两个空源、一个零权重）在四种 `(N, T)` 下与 oracle 对拍；
  另加 `[10**5] * 3`、`N = 30000` 的三条交错阶梯（`schedule[:6] == [0,1,2,0,1,2]`）
  与「源 0 的文档号恰好是 `range(10000)`」。
- `check_invalid`：39 个非法入参 + 4 个合法边界。四个边角值得单说：`10**400` 的权重
  `float()` 会 `OverflowError`、`1e200 ** 2` 也会 `OverflowError`（都必须转成
  `ValueError`）；`[1e308, 1e308]` 的和会溢出成 `inf`；`(-2.0) ** 0.5` 在 Python 里
  **返回复数**而不是报错，所以负权重必须在缩放前就拒掉。

变异测试 24 个 mutant 全部被抓：忽略温度、指数取反、空源不置零、`base` 改成四舍五入/上取整、
平局给大下标、`extra` 发给前几个下标、配额留成浮点、按配额降序放置、理想槽改成
`round`/恒为 0、向后扫描、干脆不扫描、文档号取槽位号/不取模、`flat` 丢掉偏移、
`offsets` 改用 inclusive、`bool` 当 `int`、放行负权重、不查 `total` 是否有限、
放行 0 温度、放行浮点预算、`probs` 恒等于 `1/k`。

另有两个**等价改写**不计入：`total` 反向累加、`e/total` 写成 `e * (1/total)`。它们与参考解
只差浮点最后一位，题面对 `probs` 只要求 1e-12 容差，任何 fixture 都不会因此改判——
把这类差异也判错反而是对候选人的苛求，所以题面写的是「`total` 是 `e` 各项之和」而不是
「按下标升序左结合」。

## 范围与来源

- **不做归属声明**。DoReMi / megatron 的混合权重、HuggingFace 的数据混采脚本都用了
  \(w^{1/T}\) 与整数配额，但离线环境里没有任何一份可对照的实现（`.venv-eval` 只有
  torch / numpy / transformers，没有 datasets / datatrove / megatron）。
  `expo = 1/T`、空源强制置零、「最大余额 + 平局取小下标」、「按源下标升序放置 + 理想槽
  向后扫描」、「`j % sizes` 循环」全部是**题面自定**的一套自洽口径；真实管线在这些点上
  各不相同（是否按 token 而非样本计数、是否允许同一篇重复、是否在源内再 shuffle）。
- **算法性质靠自测背书**：温度这一档先看两条应当成立的单调性——温度越高熵越高，
  并且「拿不到配额的源」会减少。20 组随机权重 \(\times\) 6 个源、`N = 100`、自然对数：

  ```text
  # 20 random weight vectors over 6 sources, N = 100, sizes = 1000 each
  # H = natural-log entropy of probs (uniform would be ln 6 = 1.7918)
  # starved = entries with counts == 0 while probs * N > 0, summed over the 20 blends
     T      mean H(probs)   max |counts - q|   starved sources   max |sum probs - 1|
     0.25        0.9967           0.635099                35            2.22e-16
     0.50        1.3482           0.584094                11            2.22e-16
     1.00        1.6032           0.650223                 1            2.22e-16
     2.00        1.7265           0.585846                 0            2.22e-16
     4.00        1.7722           0.676399                 0            2.22e-16
    16.00        1.7904           0.750774                 0            2.22e-16

  # ideal-slot forward scan, random quota vectors (N <= 300, 1..9 sources)
  5000 vectors scanned, overruns beyond the array: 0
  ```

  熵随温度单调上升并逼近 `ln 6`；`T = 0.25` 时 120 个「源×混合」组合里有 35 个被压到
  0 配额——这正是「直接按权重归一化会把小语种饿死」那句话的量化版本，也是温度存在的理由。
  `max |counts - q|` 全程 \(< 1\)，与构造一致（不随温度变小，因为它是整数分配的固有误差）。
- **与相邻题的分工**：`dt-rank-shard-indices`（549）管「一条全局索引怎么切给各 rank
  （跨步切片、从头补齐、全局覆盖账）」，本题管**切之前**「多源怎么混、每个源占几帧、
  排在第几帧」；`sequence-pack-pad`（120）管定长打包；`fw-distributed-sampler`（253）是
  单机 `DistributedSampler` 的索引账，不做多源配额。`dt-unigram-sample`（462）是分词器
  内部的 Unigram 后验路径采样，与数据混合无关——它采的是 token，本题采的是文档。
