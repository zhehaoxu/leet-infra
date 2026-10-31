# 数据集混采：温度权重、最大余额配额与平滑排布

## 题目描述

多语料、多任务、多模态训练里，「每个数据源贡献多少样本」本身就是一份需要设计的配方。手上有 `k` 个数据源，各自有篇数 `sizes[i]` 和一份原始权重 `weights[i]`，一个 step 的总样本数预算是 `num_samples`。要把它落成一条**确定性**的索引序列，中间有四道关卡：

1. **温度混采**。直接把权重归一化往往太陡——小语种被压到几乎见不到。业界通行做法是先做 \(w_i^{1/T}\) 再归一化：\(T>1\) 把分布往均匀方向压，\(T<1\) 把分布往权重最大的方向收紧，\(T=1\) 就是原始比例。这就是 DoReMi / megatron 式混合里那个 alpha。
2. **整数配额**。归一化概率乘预算一般不是整数。逐项向下取整会让总数少几个，逐项四舍五入又可能多几个甚至超过预算。**最大余额法**（largest remainder / Hare quota）是唯一同时满足「总数恰等于预算」和「每项与理想值差不到 1」的整数分配，选哪几个来源补最后一票必须有确定的平局规则。
3. **平滑排布**。配额定了「各来源取几个」，还没定「排在第几帧」。把某来源的样本全堆在 epoch 开头，梯度会在训练尾段塌到只剩一个域。平滑做法是给第 `j` 次发射算一个理想槽位，再往后找第一个空位——于是排布是纯函数，可复现、可分片。
4. **循环越采样**。小源配额可能大于它的篇数（`num_samples` 大于语料总量时必然发生），这时按「第 `j` 次发射取 `j % sizes[i]`」循环复读，而不是无放回抽干；篇数为 0 的源必须完全拿不到配额。

本题把四道关卡串成一条流水线，并要求把**五个中间结果**全部交出来：概率、配额、来源序列、源内文档号、全局文档号。任何一步理解错，都会在下游那一格里露出来。全程纯 Python 浮点与整数运算，没有随机数、没有 `random.shuffle`、没有 `WeightedRandomSampler`。

## 输入

- `sizes`：非空 `list[int]`，每个源的数据条数，`0 <= sizes[i]`。
- `weights`：`list`，长度必须与 `sizes` 相等，每项是非负、能转成有限 `float` 的 `int`/`float`。
- `num_samples`：非负整数，本 epoch 的总样本预算。
- `temperature`：正数、能转成有限 `float` 的 `int`/`float`，混采温度。

**禁止**使用任何随机数、任何采样库；四道关卡都要手写。

## 精确规则

记 `k = len(sizes)`，以下顺序是题面契约的一部分。

1. **温度权重**。指数 `expo = 1.0 / temperature`，逐源
   \[ e_i = \begin{cases} 0.0, & sizes[i] = 0\\ w_i^{\,expo}, & \text{否则}\end{cases} \]
   篇数为 0 的源**强制**置零，与它的权重无关。
2. **归一化**。`total` 是 `e` 各项之和，`probs[i] = e_i / total`（浮点比较容差 1e-12，结合顺序不计）。`total` 必须有限且 \(> 0\)，否则抛 `ValueError`。
3. **最大余额配额**。理想值 \(q_i = probs[i] \cdot num\_samples\)，`base_i = floor(q_i)`（向下取整，不是四舍五入）。令 `extra = num_samples - sum(base)`，把下标按 \((q_i - base_i)\) **降序**排列、**平局时取下标小者在前**，前 `extra` 个源各加 1。得到整数 `counts`，满足 `sum(counts) == num_samples` 且 \(|\,counts_i - q_i\,| < 1\)。
4. **平滑排布**。开一个长度 `num_samples` 的空序列。按**源下标升序**逐源放置；该源第 `j` 次发射（`0 <= j < counts[i]`，即 `j` 是该源内部的发射计数）的理想槽位为
   \[ slot_0 = \left\lfloor \frac{j \cdot num\_samples}{counts[i]} \right\rfloor \]
   从 `slot_0` 起**向后**线性扫描，落在第一个空闲槽上写入 `schedule[slot] = i`。`counts[i] = 0` 的源不占任何槽。所有发射的槽位一定存在且不越界。
5. **循环越采样与全局编号**。记某源第 `j` 次发射落在槽 `t`，则 `doc_indices[t] = j % sizes[i]`。`offsets` 是 `sizes` 的 exclusive 前缀和（`offsets[0] = 0`，`offsets[i] = sizes[0] + ... + sizes[i-1]`），`flat_indices[t] = offsets[schedule[t]] + doc_indices[t]`，即把该文档拼到全库大数组里的下标。

## 返回

五元组 `(probs, counts, schedule, doc_indices, flat_indices)`，全部是 Python 内置类型：

- `probs`：`list[float]`，长度 `k`，和为 1（浮点意义下）。
- `counts`：`list[int]`，长度 `k`，和为 `num_samples`。
- `schedule`：`list[int]`，长度 `num_samples`，值是源下标，`i` 出现 `counts[i]` 次。
- `doc_indices`：`list[int]`，长度 `num_samples`，值是**源内**文档号，`0 <= doc_indices[t] < sizes[schedule[t]]`。
- `flat_indices`：`list[int]`，长度 `num_samples`，全局文档号。

非法入参一律抛 `ValueError`：`sizes` 不是 `list` 或为空、元素不是 `int`（`bool` 不算 `int`）或为负；`weights` 不是 `list`、长度不等于 `k`、元素不是 `int`/`float`（`bool` 不算）、为负、`NaN`、`inf` 或大到无法转成 `float`；`num_samples` 不是 `int`（`bool` 不算）或为负；`temperature` 不是 `int`/`float`（`bool` 不算）、`<= 0`、`NaN`、`inf` 或大到无法转成 `float`；幂运算溢出、`total` 非有限或 `total == 0`（例如全部权重为 0，或所有非空源都拿到 0 权重）。`num_samples = 0` 是合法输入，`probs` 照常归一化，后三个列表为空。不得修改 `sizes` 或 `weights`。

## 示例

例一，温度 1 就是原始比例，`k` 个源的配额靠最大余额补齐：

```
sizes = [2, 3, 5], weights = [1, 1, 2], num_samples = 7, temperature = 1.0

expo = 1/1 = 1, e = [1, 1, 2], total = 4
probs  = [0.25, 0.25, 0.5]
q      = [1.75, 1.75, 3.5]          base = [1, 1, 3]  sum 5, extra 2
rem    = [0.75, 0.75, 0.5]          descending, tie -> smaller index first
                                    -> bump sources 0 and 1
counts = [2, 2, 3]

# place source 0 (c=2) first: ideals floor(0*7/2)=0, floor(1*7/2)=3
# then source 1 (c=2): ideal 0 busy -> 1 ; ideal 3 busy -> 4
# then source 2 (c=3): ideal 0 busy -> 2 ; ideal 2 busy -> 3 busy -> 5 ; ideal 4 busy -> 6
schedule    = [0, 1, 2, 0, 1, 2, 2]
doc_indices = [0, 0, 0, 1, 1, 1, 2]      # per-source emission counter j
offsets     = [0, 2, 5]                  # exclusive prefix sum of sizes
flat        = [0, 2, 5, 1, 3, 6, 7]
```

例二，篇数为 0 的源被强制排除，温度 2 开平方削平差距：

```
sizes = [4, 0, 9], weights = [1, 5, 4], num_samples = 5, temperature = 2.0

expo = 0.5, e = [1.0, 0.0, 2.0]     # source 1 has 0 documents -> forced to 0
                                     # (sqrt(5) = 2.236... would have been its entry)
probs  = [0.3333333333333333, 0.0, 0.6666666666666666]
q      = [1.6666666666666665, 0.0, 3.333333333333333]
counts = [2, 0, 3]                  # base [1,0,3], extra 1 goes to the largest remainder

schedule    = [0, 2, 0, 2, 2]
doc_indices = [0, 0, 1, 1, 2]
offsets     = [0, 4, 4]             # the empty source still consumes no range
flat        = [0, 4, 1, 5, 6]
```

例三，温度 0.5 把差距放大成平方，配额 7 超过篇数 5，于是循环复读：

```
sizes = [5, 5], weights = [1, 3], num_samples = 8, temperature = 0.5

expo = 2.0, e = [1.0, 9.0], probs = [0.1, 0.9]
q = [0.8, 7.2], base = [0, 7], extra = 1 -> remainder 0.8 wins
counts = [1, 7]

schedule    = [0, 1, 1, 1, 1, 1, 1, 1]
doc_indices = [0, 0, 1, 2, 3, 4, 0, 1]   # j % 5 wraps: source 1 is oversampled
offsets     = [0, 5]
flat        = [0, 5, 6, 7, 8, 9, 5, 6]   # ids 5 and 6 appear twice, by design
```

平局规则单独看一眼：四个等权源、预算 6，理想值全是 `1.5`，`base` 全 1、`extra` 2，
两票都按「下标小者在前」发出去，所以 `counts == [2, 2, 1, 1]` 而**不是** `[1, 1, 2, 2]`。
