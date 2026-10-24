# 解题说明 · Lookahead 的 Jacobi n-gram 池

## 把 draft 换成"查历史 n-gram"

Lookahead 不设 draft 模型，而是把语言建模视作一组可以 Jacobi 并行迭代的方程：猜测若干连续位置，用**已见过的 n-gram 续写**来填充这些猜测。落到单流实现，核心就是一个 `pool: n-gram -> 下一 token` 的表：
- 预置阶段从 `pool_seed` 序列里，把每个长度 $\in[\text{min\_n},\text{max\_n}]$ 的子串映射到它的下一个 token；
- 生成时用当前 $seq$ 的**最长后缀**去查表，命中即投机得到 `drafted`；
- 查不到就退回一个确定性 token $(seq_{-1}+1)\bmod V$（真实实现里这一步是 target 模型的一次前向，这里用确定性函数替代以便判题）。

## 边生成边学习

每追加一个 token，就把"它前面那段 n-gram → 它"写回 pool，于是池随生成不断扩充、自我强化。两个顺序细节必须固定：
1. **后缀查找**从 `max_n` 递减到 `min_n`——优先用更长上下文，避免短 n-gram 抢命中；
2. **池更新**从 `min_n` 递增到 `max_n`——因为一次更新会写多个窗长的键，顺序决定了覆盖结果，必须与 oracle 完全一致。

还有**先查后写**：本步的草稿只依赖写入前的池，保证同一 token 不会既当查询上下文又影响自身草稿。

## 与其它投机题的区别

Lookahead / REST / n-gram 都是"检索式草稿"（不用神经网络 draft）。Lookahead 用**定长窗后缀 + 自学习池 + 确定性回退**；REST（题 180）则用**变长最长后缀匹配**且不做逐 token 回退。二者都强调无 draft 模型的轻量投机。

## 边界自检

- `check_example`：定长 2-gram 池，先两次命中、后两次回退（链 `[0,1,2,3]` → `[2,3,4,5]`）。
- `check_empty_pool_fallback`：空池 → 全走 $(last+1)\bmod V$ 回退。
- `check_seed_override`：同名 n-gram 被后一条 seed 覆盖，验证"后来者覆盖"。
- `check_matches_oracle`：随机 prompt/窗宽/seed/词表，与独立纯 Python 实现对拍逐步一致。
