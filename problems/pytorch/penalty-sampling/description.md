# 重复惩罚采样 (Penalties)

## 题目描述

模型一旦进入「复读机」模式就会反复吐出同样的词。llama.cpp 的 `penalties` 采样器把三种
反重复惩罚合成一步，直接**篡改 logits**（在 softmax 之前），压低那些最近出现过的 token：

- **penalty_repeat**（乘法，经典 XPRisk / frequency penalty）：对窗口内出现过的 token，
  正 logit 除以 `repeat`、负 logit 乘以 `repeat`，效果都是把它**往更小推**。`repeat=1.0` 即关闭。
- **penalty_freq**（加法）：再减去 `count * freq`——出现次数越多，罚得越重。
- **penalty_present**（加法）：再减去 `(count>0) * present`——只要出现过就罚一次，与次数无关。

三者叠加。与前面各题不同，这一题**不做采样**，只输出惩罚后的 logits 向量。

## 规则

- 实现 `solve(logits, history, penalty_repeat, penalty_freq, penalty_present, penalty_last_n)`：
  - `logits`：形状 `(V,)` 的一维原始 logits。
  - `history`：最近生成的 token 索引列表（**时间正序**，末尾是最新的一个）。
  - `penalty_last_n`：统计窗口大小——只看 `history` 末尾的 `penalty_last_n` 个 token；
    传 `-1` 表示整段 history 都在窗口内。
- 步骤：① 取窗口 `win = history if penalty_last_n < 0 else history[-penalty_last_n:]`；
  ② 统计 `win` 中每个 token 的出现次数 `count`；③ 对每个 `count>0` 的 token `t`（`l = logits[t]`）：
  $$l \leftarrow \begin{cases} l \cdot \text{repeat}, & l \le 0\\[2pt] l \,/\, \text{repeat}, & l > 0\end{cases}
  \qquad l \leftarrow l - \text{count}\cdot\text{freq} - (\text{count}>0)\cdot\text{present}$$
- 未被惩罚的 token 原样保留。返回**惩罚后**的 `logits`（形状 `(V,)`）。

## 示例

```
输入：logits = [1, -2, 0.5, 3, -0.5]，history = [0, 1, 0, 2, 0]，
      penalty_repeat = 1.1，penalty_freq = 0.1，penalty_present = 0.2，penalty_last_n = -1
  窗口计数 count = {0:3, 1:1, 2:1}
  token0 (l=1.0>0): 1.0/1.1 = 0.909091；− 3*0.1 − 0.2 = −0.5 → 0.409091
  token1 (l=-2.0≤0): -2.0*1.1 = -2.2；− 1*0.1 − 0.2 = −0.3 → -2.5
  token2 (l=0.5>0): 0.5/1.1 = 0.454545；− 0.1 − 0.2 = −0.3 → 0.154545
  token3、token4 未出现 → 不变 (3.0, -0.5)
输出：penalized = [0.409091, -2.5, 0.154545, 3.0, -0.5]
```

## 约束

- float32 输入；`history` 中的索引都在 `[0, V)` 内
- 惩罚顺序固定：**先** repeat 乘/除、**后** 减 `count*freq + present`
- `penalty_repeat=1.0, penalty_freq=0.0, penalty_present=0.0` 时输出恒等于输入
- 返回一维张量，形状与 `logits` 相同
