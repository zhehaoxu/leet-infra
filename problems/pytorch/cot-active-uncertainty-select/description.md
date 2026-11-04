# Active-Prompt 不确定度选样

## 题意

CoT demo 的人工标注很贵。Active-Prompt（arXiv:2302.12246）用**不确定度**挑出最值得标注的问题：同一道题让模型采样 k 次，答案五花八门的题说明模型还没学会，优先送人工标注；答案高度一致的题跳过。这就是 active learning 的「查询不确定度」在 prompt 工程里的落地。

给你（GPT 采样与答案抽取在池外完成，`retrieval-kmeans-lloyd` 无关、本题无聚类）：`samples_per_question`——每题 k 个**抽取后**答案串（空串表示该次无解）；以及三个口径开关 `sort_by` / `dataset_kind` / `qes_limit`。

## 你要实现的账本

### 1. occurrence 计数

逐题逐采样：非空答案按**原串**为键计数（dict 按首现序）；空答案计入哨兵键 `NO_SOLUTION = "-10086"`。`strategyqa` 的 occurrence **预置** `{"yes": 0, "no": 0}` 两键。

### 2. variance（仅 numerical）

```
ans_list = [float(ans) for ans, c in occurrence.items() for _ in range(c)]
variance = 总体方差（ddof=0）
```

**NO_SOLUTION 也以 -10086 计入方差**——官方原样，一道题全是无解时方差巨大。

### 3. entropy（scipy.stats.entropy 口径）

```
entropy = -Σ_{c>0} (c/n)·ln(c/n)      # 0 频率项贡献 0
```

频率全零（n=0）时为 nan。判题环境无 scipy，自实现对齐：`entropy([2,2]) = ln2`、单值分布为 0。

### 4. disagreement

`len(occurrence)`——distinct 答案数（含 NO_SOLUTION 键；strategyqa 的预置 yes/no 恒算两键）。

### 5. 排序（stable，平局保持输入序）

| sort_by | 键 | 说明 |
|---|---|---|
| disagreement | `-len(occurrence)` 降序 | distinct 答案多者在前 |
| disagreement + strategyqa | `abs(yes - no)` **升序** | 两票越接近越不确定；官方 except 兜底属死分支 |
| variance | `-variance` 降序 | **仅 numerical**；官方 elif 带 dataset 过滤，不匹配时静默不排序 |
| entropy | `-entropy` 降序 | 全任务 |

### 6. qes_limit

只计算**前 qes_limit 题**（0 = 全部；超过题数即全部）；排序只作用于被计算的子集。

返回 records 列表（排序后），每条 `dict(dataset_idx, variance?, entropy, occurrence, disagreement)`，`dataset_idx` 为原输入下标。

## 输入约束

- `samples_per_question` 非空 list[list[str]]，每题子列表非空（num_trails ≥ 1）；
- `sort_by` ∈ {disagreement, variance, entropy}；`dataset_kind` ∈ {generic, numerical, strategyqa}；
- `qes_limit` ≥ 0 的整数（bool 不算数）。

## 提示

- python 的 `sort(key=..., reverse=True)` 平局**保持原序**（不是反转），与 `-key` 降序恒等；
- `qes_limit == n` 时 cap 与否结果相同；
- 熵的 0 频率项（如 strategyqa 全投 yes 时预置的 no:0）对熵无贡献，但**算进 disagreement**。
