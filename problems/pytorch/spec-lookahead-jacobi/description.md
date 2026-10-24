# Lookahead 解码：Jacobi n-gram 候选池

## 题目描述

**Lookahead Decoding**（Fu et al., 2024, [arXiv:2402.02057](https://arxiv.org/abs/2402.02057)）不需要独立 draft 模型，而是把解码看成求解一组 Jacobi 迭代：并行地维护一批 n-gram 候选（**pool**），用当前上下文的最长后缀去命中池中已见过的续写来"投机"，命中失败就回退，同时把新生成的 n-gram→后继 关系写回池，逐步自学习。本题让你实现这个 **pool 匹配 + 确定性回退 + 池更新** 的核心循环（抽掉并行 n-gram 分支的细节，聚焦单流的候选检索与学习）。

## 规则

入参：
- `prompt`：初始已生成序列 `list[int]`。
- `steps`：要生成的 token 数。
- `min_n`、`max_n`：n-gram 窗长下/上界。
- `pool_seed`：预置池的若干 token 序列 `list[list[int]]`。
- `V`：词表大小（回退取模用）。

**预置 pool**：对每条 seed 序列 $s$，对每个起点 $i$ 与每个 $n\in[\text{min\_n},\text{max\_n}]$，若 $i+n<|s|$，令
$$\text{pool}[(s_i,\dots,s_{i+n-1})] = s_{i+n}.$$
按 seed 顺序与 $(i,n)$ 顺序写入，**后来者覆盖**同名 n-gram。

**每一步生成**（共 `steps` 步，$seq$ 为当前序列）：

1. **查草稿**：$n$ 从 $\text{max\_n}$ 递减到 $\text{min\_n}$，若 $|seq|\ge n$ 且 $(seq_{-n},\dots,seq_{-1})\in\text{pool}$，取 `drafted = pool[key]` 并停止查找（**最长后缀优先**）。
2. **回退**：若没有任何后缀命中，$drafted=(seq_{-1}+1)\bmod V$（确定性）。
3. 把 $drafted$ 追加到 $seq$，记入输出。
4. **更新池**：对每个 $n\in[\text{min\_n},\text{max\_n}]$，若 $|seq|\ge n+1$，令 $\text{pool}[(seq_{-(n+1)},\dots,seq_{-2})]=seq_{-1}$（把紧邻新 token 之前的 $n$-gram 映射到新 token）。

返回 `torch.long` 一维张量，长度恰为 `steps`。

- "最长后缀优先"体现 look-ahead：能用更长的已见上下文就用，草稿更靠谱。
- 池是**有状态**的、边生成边学；同一步里先查后写，保证与独立 oracle 逐步一致。

## 示例

```
prompt=[0,1], steps=4, min_n=2, max_n=2, pool_seed=[[0,1,2,3]], V=10
预置 pool = {(0,1):2, (1,2):3}

step1 seq=[0,1]        后缀 (0,1) 命中 -> 2   ；更新 pool[(0,1)]=2
step2 seq=[0,1,2]      后缀 (1,2) 命中 -> 3   ；更新 pool[(1,2)]=3
step3 seq=[0,1,2,3]    后缀 (2,3) 未命中 -> 回退 (3+1)%10 = 4 ；更新 pool[(2,3)]=4
step4 seq=[0,1,2,3,4]  后缀 (3,4) 未命中 -> 回退 (4+1)%10 = 5
输出 = [2, 3, 4, 5]
```

## 约束

- CPU；返回长度 `steps` 的 `torch.long`，用 `torch.equal` 比较。
- token id 均落在 $[0,V)$；回退严格按 $(seq_{-1}+1)\bmod V$。
- 后缀查找从 `max_n` 到 `min_n`；池更新从 `min_n` 到 `max_n`（顺序影响覆盖结果，须与题面一致）。
