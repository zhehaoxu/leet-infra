# DRY 采样 (Dont Repeat Yourself)

## 题目描述

普通重复惩罚只看「单个 token 出现过几次」，管不住「一整段 n-gram 反复循环」
（`...ab ab ab ab...`）。**DRY**（impl/llama.cpp 的 `dry`）惩罚的是**重复的序列**：它找出
「上下文末尾那段重复子串」，对**会把循环再延长一步的那个 token**施加以**指数级**增长的惩罚
（重复越长，罚得越狠：$\text{mult}\cdot\text{base}^{\text{重复长}-\text{allowed}}$）。

llama.cpp 用**反向 Z-algorithm** 在线性时间内算出每个位置的「后缀重复长度」。本题实现其
**不含 sequence breakers** 的版本（不做重启串截断，`rep_limit = N`）。与 penalty 题一样，
DRY **只改 logits、不做采样**。

## 记号

- `last = history`（若 `dry_penalty_last_n < 0`）或 `history[-dry_penalty_last_n:]`；$N = \text{len}(last)$。
- `rev = last` 的逆序：$\text{rev}[j]$ 即「从末尾数第 $j$ 个 token」（等价 llama.cpp 的 `rat(j)`）。
- $L_k$（`repeat_len` at shift $k$）：`rev` 与 `rev[k:]` 的**最长公共前缀**长度，即最大的 $n$ 使
  $\text{rev}[0\!:\!n]=\text{rev}[k\!:\!k+n]$（可用朴素 while 逐个比，也可用 Z-algorithm 加速，结果一致）。

## 规则

- 实现 `solve(logits, history, dry_multiplier, dry_base, dry_allowed_length, dry_penalty_last_n)`：
  `logits` 形状 `(V,)`，`history` 为时间正序的 token 索引列表。
- 若 `dry_multiplier == 0`、`dry_base < 1`、`dry_penalty_last_n == 0`，或 $N \le$ `dry_allowed_length`：
  直接返回原 `logits`（无任何惩罚）。
- 否则：
  1. 对每个 shift $k = 1, \dots, N-1$ 计算 $L_k$（如上）。
  2. 若 $L_k \ge$ `dry_allowed_length`，则**候选惩罚 token** 为 $\text{last}[N-k]$，其重复长度为 $L_k$；
     同一个 token 取所有命中里**最大**的 $L_k$。
  3. 对每个被惩罚 token $t$、其最大重复长 $R$：
     $$\text{logits}[t] \leftarrow \text{logits}[t] - \text{dry\_multiplier}\cdot \text{dry\_base}^{\,\max(R-\text{allowed},\,0)}$$
     （防溢出：记 `max_exponent = int(88.7228391 / log(dry_base))`（当 `dry_base > 1.000001` 时），
     指数超过它则截断到 `max_exponent`。）
- 未被惩罚的 token 原样保留。返回惩罚后的 `logits`（形状 `(V,)`）。

> 直觉：$\text{last}[N-k]$ 是「重复段起点往前一格」的那个 token——正是它的下一次出现会把这段
> 循环续得更长，所以要罚它。

## 示例

```
输入：logits = zeros(8)，history = [5,6,7,5,6,7]，mult = 1.0，base = 1.5，allowed = 2，last_n = -1
  N = 6，rev = [7,6,5,7,6,5]
  L_1 = 0（7≠6）; L_2 = 1（7=7 后 6≠7）; L_3 = 3（[7,6,5]=[7,6,5]）; L_4 = 0; L_5 = 0
  仅 k=3 满足 L_3=3 ≥ allowed=2 → 惩罚 token = last[6-3] = last[3] = 5，重复长 R = 3
  指数 = R - allowed = 1，惩罚 = 1.0 * 1.5^1 = 1.5
输出：penalized = [0, 0, 0, 0, 0, -1.5, 0, 0]   （索引 5 = -1.5）
```

```
输入：history = [0,1,0,1,0,1]，其余同上（mult=1.0 base=1.5 allowed=2 last_n=-1）
  rev = [1,0,1,0,1,0]；L_2 = 4（前缀 [1,0,1,0] 命中）
  惩罚 token = last[6-2] = last[4] = 0，重复长 R = 4
  指数 = 4 - 2 = 2，惩罚 = 1.0 * 1.5^2 = 2.25
输出：penalized = [-2.25, 0, 0, 0, 0, 0, 0, 0]   （索引 0 = -2.25）
```

## 约束

- float32 输入；`history` 索引在 `[0, V)` 内；本题不含 sequence breakers
- `logits[t] -= mult * base**(R-allowed)` 是唯一惩罚动作；重复长不同 → 惩罚呈 `base` 的幂差
- 同一 token 被多个 shift 命中时取**最大**重复长（幂最高）
- 触发早退（原样返回）的四种情形见「规则」
