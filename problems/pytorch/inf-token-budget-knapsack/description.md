# 每步 token 预算：prefill 准入的 0/1 背包与余量分块

## 题目描述

一次 forward 能处理的 token 数是固定的，这是 serving 引擎最硬的预算：decode 请求每个至少要吃 1 个名额（否则这一路输出就断了），剩下的名额才能拿去 prefill 新请求。而新请求的 prompt 长短不一，优先级也不一样——一步之内把哪些请求**整段** prefill 完，是一道带权背包；背包装不满的零头，通常再切一刀给队列里最靠前的那个请求，让它也往前走一点（chunked prefill 的渐进准入）。

本题把这一步的预算决策完整算出来：给定等待队列的 prompt 长度与优先级权重、本步预算与 decode 请求数，输出入选集合、总权重、余量分块落在谁身上、分块多少、还剩多少没用掉。

## 规则

所有量都是整数。一步之内：

1. **decode 保底**：先扣掉 `decode_count` 个名额，剩 `remaining = budget - decode_count` 个名额留给 prefill（题目保证 `budget >= decode_count`）。
2. **0/1 背包**：选一个下标集合 `S`，要求 `Σ prompt_lens[i] <= remaining`（`i ∈ S`），被选中的请求这一步「整段 prefill 完成」。目标按字典序依次比较：
   - `Σ values[i]` 最大；
   - 并列时 `Σ prompt_lens[i]` 最小（省下的名额留给后面）；
   - 再并列时，`S` 的升序下标列表按字典序最小。
3. **余量分块**：记 `leftover = remaining - Σ_{i∈S} prompt_lens[i]`。若 `leftover > 0` 且存在未入选请求，取**下标最小**（FCFS 最靠前）的未入选请求 `j`，给它分 `min(leftover, prompt_lens[j])` 个 token 的分块——它本步不算完成、也不算入选；分块之后剩下的 token 记作 `unused`。

返回五元组 `(served, total_value, chunk_index, chunk_tokens, unused)`：

- `served`：入选集合的升序下标列表；
- `total_value`：入选集合的权重和；
- `chunk_index`：拿到余量分块的下标，没有分块时为 `-1`；
- `chunk_tokens`：分块 token 数，没有分块时为 `0`；
- `unused`：分块之后仍未用掉的 token 数。

注意权重为 0 的请求不会入选（入选只会白占名额），但它照样可能拿到余量分块。

## 示例

**示例 1**：`prompt_lens = [5, 3, 8, 3]`、`values = [4, 2, 9, 1]`、`budget = 10`、`decode_count = 2`

```
remaining = 10 - 2 = 8
候选集合（占用 <= 8）：
  {0, 1} 占用 8  权重 6
  {0, 3} 占用 8  权重 5
  {1, 3} 占用 6  权重 3
  {2}    占用 8  权重 9     <- 权重最大
  ...
served = [2]，正好用满 8 个名额，leftover = 0
```

```
served       = [2]
total_value  = 9
chunk_index  = -1
chunk_tokens = 0
unused       = 0
```

**示例 2**（0 权重与未用尽的余量）：`prompt_lens = [1, 2, 6]`、`values = [10, 0, 1]`、`budget = 10`、`decode_count = 0`

```
remaining = 10
{0, 2}     占用 7  权重 11   <- 最优
{0, 1, 2}  占用 9  权重 11   <- 权重相同但占用更大，输在第二关键字
leftover = 10 - 7 = 3
未入选的里下标最小的是 1（prompt=2）-> 分块 min(3, 2) = 2
unused = 3 - 2 = 1
```

```
served       = [0, 2]
total_value  = 11
chunk_index  = 1
chunk_tokens = 2
unused       = 1
```

**示例 3**（并列时省名额）：`prompt_lens = [2, 4]`、`values = [5, 5]`、`budget = 4`、`decode_count = 0`

```
remaining = 4
{0} 占用 2 权重 5；{1} 占用 4 权重 5  -> 权重并列，取占用更小的 {0}
leftover = 4 - 2 = 2，分块给未入选的下标 1：min(2, 4) = 2
```

```
served       = [0]
total_value  = 5
chunk_index  = 1
chunk_tokens = 2
unused       = 0
```

## 约束

- 请求数不超过 12；`prompt_lens[i] >= 1`；`values[i] >= 0`；`budget >= 1`；`0 <= decode_count <= budget`；`budget <= 128`。
- 每步只分一个余量分块，不会连续喂给多个请求。
- 返回的 `served` 必须是升序去重下标，其余四项都是 Python `int`。
- 判题只比较返回值，不限定实现（容量 DP、记忆化搜索、暴力枚举子集都可以）。
