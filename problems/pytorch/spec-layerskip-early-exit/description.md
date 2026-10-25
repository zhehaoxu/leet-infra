# LayerSkip：逐层早退草稿与剩余层验证的调度核算

## 题目描述

**LayerSkip**（Elhoushi & Shrivastava et al., 2024, [arXiv:2404.16710](https://arxiv.org/abs/2404.16710)，[facebookresearch/LayerSkip](https://github.com/facebookresearch/LayerSkip)）是一种**自推测解码**：不要独立的 draft 模型，而是让同一个模型「浅一点退出」来起草、再「把剩下的层跑完」来验证。训练侧用 layer dropout + early-exit loss 把模型训成一堆不同深度的子模型的集成，所有层共用同一个 LM head；推理侧草稿阶段只跑前 $E$ 个 transformer block 就直接反嵌入出 token，验证阶段则借助 KVQ cache（KV cache 加上退出层的 query）**只跑剩下 $L-E$ 个 block**，从而让 draft 与 verify 共享算力。原文的接受判据很朴素：把完整模型对每个草稿位给出的 token 与草稿 token 逐个比对，disagreement 之前的都收下，再补上 disagreement 处（或全部一致时多出的那个）目标 token。

本题把这套调度做成一个**确定性模拟**：给定玩具模型在每个草稿位、每个退出深度上的输出分布，以及完整模型的目标分布，让你决定每个草稿位在哪一层早退、验证阶段要跑几层，并核算这一轮**产出了多少 token、执行了多少个 block、整体加速比是多少**。注意草稿深度是**逐 token 自适应**的（LayerSkip 原文按 token 难度在不同层退出，本题用一个置信阈值把它形式化），而验证是一次并行前向，只能复用所有草稿位**共同拥有**的那段浅层缓存。

## 规则

入参：
- `Q`：float $(R, \gamma, D, V)$。`Q[r][j][l]` 是请求 $r$ 的第 $j$ 个草稿位在**跑完第 $l+1$ 个 block 后**用共享 LM head 反嵌入出的分布（每行和为 1，$D \le L$）。
- `P`：float $(R, \gamma, V)$。`P[r][j]` 是完整模型（全部 $L$ 层）在同一位置上的目标分布。
- `tau`：float，早退置信阈值。
- `L`：int，模型总 block 数（$1 \le D \le L$）。

每个请求独立地模拟一轮「$\gamma$ 个草稿位 + 一次并行验证」：

1. **早退出草稿**：草稿位 $j$ 从第 1 个 block 开始逐层加深，取**第一个**满足 $\max_v \texttt{Q[r][j][l][v]} \ge \texttt{tau}$ 的层号 $l$，早退深度 $e_j = l + 1$；若 $D$ 层都不达标则 $e_j = D$。草稿 token $d_j = \arg\max_v \texttt{Q[r][j][}e_j\texttt{-1][}v]$（并列取最小下标）。草稿阶段代价 $=\sum_j e_j$ 个 block。
2. **剩余层验证**：一次并行前向只需跑草稿缓存没覆盖到的层，代价 $=\max(L-\min_j e_j,\,1)$ 个 block（所有草稿位共同跑到的最深公共前缀是 $\min_j e_j$；若已在最后一层退出，收尾 token 仍需一次 LM head，按 1 个 block 计入）。
3. **贪心一致判定**：记 $t_j = \arg\max_v \texttt{P[r][j][}v]$（并列取最小下标），从 $j=0$ 起，只要 $d_j = t_j$ 就接受该草稿位；第一次不一致立刻停止，该位置改由目标 token $t_j$ 顶上。
4. **本轮产出** $\;$ token 数 $=$ 接受的草稿位数 $a + 1$（$a < \gamma$ 时补的是 disagreement 处的目标 token，$a = \gamma$ 时补的是全模型多算出的 bonus token）。

返回四元组 `(toks, lays, depth, speedup)`：
- `toks`：long $(R,)$，每个请求本轮产出的 token 数 $a+1$；
- `lays`：long $(R,)$，每个请求本轮实际执行的 block 数 $\sum_j e_j + \max(L-\min_j e_j, 1)$；
- `depth`：long $(R, \gamma)$，每个草稿位的早退深度 $e_j$；
- `speedup`：float，整体加速比 $= \texttt{sum(toks)} \cdot L / \texttt{sum(lays)}$。

$\gamma = 0$（`Q` 形状 $(R,0,D,V)$）表示这一轮不投机，退化为逐 token 全模型解码：`toks` 全为 $1$、`lays` 全为 $L$、`depth` 形状 $(R,0)$。

## 示例

```
R=2, gamma=3, D=3, L=5, V=4, tau=0.625（下面所有分布都是 1/16 的整数倍）

请求 0 的三个草稿位，逐层 max：
 pos0: [0.3125, 0.75, 0.5]    -> 第 2 层达标 -> e0 = 2, d0 = argmax 第2层 = 1
 pos1: [0.5, 0.5, 0.375]      -> 全不达标   -> e1 = 3, d1 = argmax 第3层 = 1
 pos2: [0.8125, 0.3125, 0.25]   -> 第 1 层达标 -> e2 = 1, d2 = argmax 第1层 = 2
 草稿代价 = 2+3+1 = 6；公共缓存前缀 min e = 1 -> 验证代价 = max(5-1, 1) = 4
 目标 argmax: P[0] = [1, 1, 1]
 j=0: 1==1 接受；j=1: 1==1 接受；j=2: 1!=2 停止 -> a=2
 toks[0] = 3, lays[0] = 6 + 4 = 10

请求 1：
 pos0: [0.3125, 0.3125, 0.5625] -> e0 = 3, d0 = 2
 pos1: [0.75, 0.375, 0.25]       -> e1 = 1, d1 = 1
 pos2: [0.375, 0.625, 0.375]    -> e2 = 2, d2 = 2   （0.625 >= 0.625 恰好踩线达标）
 草稿代价 = 3+1+2 = 6；min e = 1 -> 验证代价 = 4
 目标 argmax: P[1] = [2, 3, 0]
 j=0: 2==2 接受；j=1: 3!=1 停止 -> a=1
 toks[1] = 2, lays[1] = 6 + 4 = 10

batch: sum(toks)=5, sum(lays)=20 -> speedup = 5*5/20 = 1.25
输出 toks=[3, 2]  lays=[10, 10]  depth=[[2,3,1],[3,1,2]]  speedup=1.25

同一份调度若把模型换成 L=3（D 仍是 3）：验证代价 max(3-1,1)=2，
lays=[8, 8] -> speedup = 5*3/16 = 0.9375 < 1（模型太浅时早退省下的层不够抵验证开销）
```

## 约束

- CPU float32；$R \ge 1$，$\gamma \ge 0$，$1 \le D \le L \le 64$，$V \ge 2$；`Q` 的每一行、`P` 的每一行都和为 1，`tau` $\in [0,1]$。
- 两个 argmax（草稿 token 与目标 token）并列时一律取**最小下标**；早退判据是 $\ge$（踩线即达标）。
- `toks`、`lays`、`depth` 都是 `long` 张量，用 `torch.equal` 比较；`speedup` 是 python float，用 `allclose` 比较。
- `speedup` 是 **batch 级 token 加权**口径 $\texttt{sum(toks}\texttt{)}\cdot L / \texttt{sum(lays)}$，**不等于**各请求 $\texttt{toks}[r]\cdot L / \texttt{lays}[r]$ 的算术平均。
- 接受判据是 LayerSkip 原文的 **draft 与 target argmax 一致**：与 `inf-spec-decoding-rejection`（155，带接受概率 $\min(1,p/q)$ 与残差重采样的拒绝采样）、`spec-typical-acceptance`（178，只看 $p \ge \varepsilon$ 的阈值判据）都不同，本题完全确定、不需要任何随机数。
- 本题只管**层级调度与代价核算**，不做树形草稿：草稿树本身的最优构造与期望收益见 `spec-sequoia-optimal-tree`（177）与 `spec-sequoia-depth-bounded-dp`（283），树上逐节点验证见 `spec-specinfer-token-tree-verify`（285）。
