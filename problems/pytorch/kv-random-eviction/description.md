# KV Cache：Random Eviction 随机驱逐基线

## 题目描述

kvpress 在全部花哨的 press 之外保留了一个最朴素的 `RandomPress`（`kvpress/presses/random_press.py`）：给每个 KV 对发一个均匀随机分数做 top-k，作为所有方法的 sanity 下界——**任何打分器若跑不赢"掷骰子"，就说明它的信号是噪声**。本题把随机基线与 StreamingLLM 的头尾保底（order 145）合成标准评测形态：sink 位与最近窗口**永远保留**，只有中段按随机抽签采样；并要求顺手报告**注意力覆盖率**（保留位吃到的最后一问 attention 质量占比），让结果不止是一串下标、而能直接衡量信息损失。

判题环境禁止隐藏随机性：**每个位置的均匀随机数作为输入张量直接给出**，整个"随机"过程确定、可复现、可调试。

## 规则

输入：`draws` 形状 \((L,)\)、取值 \([0,1)\) 的均匀随机数（每位置一个）；`attn_weights` 形状 \((L,)\)、非负的最后一问对各 key 的注意力；整数 `n_sink`、`window_size`；标量 `compression_ratio` \(r\in[0,1)\)。

1. 强制保留集 \(F\) = 前 \(\min(n\_sink,L)\) 个位置 \(\cup\) 后 \(\min(window\_size,L)\) 个位置（两区间重叠时按并集去重）。
2. 总预算 \(n\_kept=\max(1,\lfloor L(1-r)\rfloor)\)；随机名额 \(q=\max(0,\ n\_kept-|F|)\)。
3. 中段 \(M=\{0,\dots,L-1\}\setminus F\)：按 `draws` **降序**取前 \(q\) 个（平局取位置小者）；若 \(|M|<q\) 则全取。
4. `kept` = \(F\cup\) 选中位，**升序**输出（int64）。注意：当 \(|F|>n\_kept\) 时保留数可以超过预算——头尾保护优先于预算。
5. 覆盖率 \(\text{coverage}=\sum_{j\in kept} a_j \big/ \sum_j a_j\)；若分母为 0，退化为 \(|kept|/L\)。

返回二元组：`kept`（int64 一维张量，升序）与 `coverage`（Python float）。

## 示例

```
L=8, n_sink=2, window_size=2, r=0.375 → n_kept = max(1,⌊8×0.625⌋) = 5
draws        = [0.6, 0.1, 0.8, 0.3, 0.9, 0.4, 0.05, 0.7]
attn_weights = [0.30, 0.20, 0.10, 0.05, 0.15, 0.05, 0.08, 0.07]

强制 F = {0,1} ∪ {6,7}，|F|=4，名额 q = 5-4 = 1
中段 M = {2,3,4,5}，draws 最大者是位置 4（0.9）→ 选中 {4}
kept     = [0, 1, 4, 6, 7]
coverage = (0.30+0.20+0.15+0.08+0.07) / 1.00 = 0.8
```

## 约束

- CPU float32；`draws` \(\in[0,1)\)；`attn_weights` 非负；\(0\le r<1\)；\(L\ge1\)。
- \(n\_sink+window\_size\ge L\) 时中段为空，全部保留，coverage = 1（只要 \(\sum a_j>0\)）。
- 平局（`draws` 相等）一律位置小者优先——测试里可能出现全相等的 `draws`。
- `kept` 升序；`coverage` 用 float 返回。
