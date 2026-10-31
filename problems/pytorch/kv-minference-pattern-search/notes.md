# 解题说明 · MInference 稀疏 pattern 搜索

## 一句话

把"这个 head 该用哪种稀疏形状"变成一道**打分题**：用一次真实的因果注意力 \(P\) 去量每个候选掩码保住了多少概率质量，逐 head 取 `argmax`。

## 质量为什么能当目标

\(P\) 每行是一个概率分布，\(\sum_j P_{ij}M_{ij}\) 就是第 \(i\) 行被 \(M\) 保住的比例，对行取平均得到一个 \([0,1]\) 的数。它不用写任何 attention kernel、也不用真跑稀疏 attention，只要掩码构造得出来就能打分——这正是上游"搜一遍就把 pattern 定下来"能成立的原因。

三种掩码的来路分别是：

- `stream_llm`：纯解析，前 \(a\) 列并上最近 \(b+1\) 条对角线，与 head 无关；
- `vertical_and_slash`：拿最后 64 个 query 的 \(P\) 当**代理**——列质量 `vertical[j]` 是窗内按列求和，对角线质量 `slash` 的第 \(j\) 项对应斜率 \(\delta=j+1-N\)；谁质量高就留谁；
- `block_sparse`：连 \(P\) 都不问，直接用块均值 \(q\)、\(k\) 做一次 \((N/32)^2\) 的块级点积来估哪块重要。

## 易错口径

1. **强制保留写成 \(+\infty\) 再去排序**。`vertical[:30]`、`slash[-30:]` 被置 \(+\infty\)，于是 top-k 丢不到它们；代价是这 30 个条目**彼此并列**：`a < 30` 时第 0 列最先被丢，`b < 30` 时最老的 \(\delta=-29\) 最先被取。若改成"前 30 个直接跳过、只排剩下的"，\(a<30\) 时要丢的列数超过 34 个非强制列，落点就变成真实 `vertical` 最小的那一列；测试 `check_vertical_force_keeps_first_30` 断言的正是掉第 0 列。
2. **块均值分母是 32，不是真实条数**。补零的 0 进了 `mean`，最后一个不满块的块向量被按比例压小，块分数随之下降。测试用 \(N=250\)（末块只有 26 行）造出两种口径选出不同块的分歧。
3. **slash 是并集**。vertical 落选的列只要落在被选中的对角线上就仍然保留；写成"与"会得到另一张掩码，\(P\) 的质量账也随之改变。
4. **名额算法是"丢最低的"**。`topk(-vertical, N - a)`、`topk(-qk, block_num - block_num//ratio)` 都是先算丢弃数；`block_num < ratio` 时丢弃数等于全部块数。

## 判分点

- `scores` 逐 head、逐候选，\(C=\texttt{len(candidates)}\)，dtype 跟输入；
- `best` 是行内 `argmax`，**同分取更小的候选下标**（测试用两个都是 1.0 的候选钉死这条）；
- `pattern` 只放尺寸、不放 kind：`block_sparse` 胜出 → `(1000, 6096)`，`stream_llm` 胜出 → 原样 `(a, b)`；
- `mask` 必须是该 head **胜出那个**候选的掩码，`(H, N, N)` bool；
- \(P\) 与全部中间量走 float64，`q`/`k` 不被就地修改，非连续视图也要能吃。

## 边界

- `N < 64` 直接报错：vertical/slash 的统计窗固定是最后 64 个 query。
- `a`、`b` 的端点：`vertical_and_slash(0, 0)` 掩码全空、分数 0，`stream_llm(0, 0)` 只剩主对角线；`a = N` 时一列不丢，掩码就是因果下三角。
- `block_sparse`：`topk_ratio = 1` 不丢块，退化成纯因果（分数 1）；`block_num < topk_ratio` 全丢（分数 0）。
- 整行被掩掉不重归一化：`a=29` 配 `b=1` 时第 0 行全空（它只能看第 0 列，而那条唯一的对角线是 \(\delta=-29\)），这一行就是往平均分里投一个 0。

## 复杂度

\(P\) 自己是 \(O(HN^2D)\) 时间、\(O(HN^2)\) 显存，搜索阶段并不省钱——上游把这段放在 `is_search` 分支里逐 head 跑一遍、结果落成 config 文件（L555-558 与 L570），之后的长上下文推理才按选中的 pattern 省。相对地，vertical/slash 统计是 \(O(H\cdot 64\cdot N)\)，`block_sparse` 的块打分只有 \(O(H(N/32)^2D)\)，每个候选的求分是 \(O(HN^2)\)。

## 出处

`microsoft/MInference`（arXiv 2407.02490）`minference/modules/minference_forward.py::search_pattern`（L129）：

- **vertical_and_slash**：L134 `last_q = 64`；L136-138 只对最后 64 个 query 取 `qk = matmul(q[…], k.transpose(2, 3))/math.sqrt(head_dim) + attention_mask[…]` 再 softmax；L139 `vertical = qk.sum(-2, keepdim=True)`、L140 `vertical[...,:30] = torch.inf`、L141 `torch.topk(-vertical, q_len - vertical_size)`；L143 `slash = sum_all_diagonal_matrix(qk)[...,:-last_q + 1]`（helper L110-116 用 `as_strided` 求对角线和）、L144 `slash[...,-30:] = torch.inf`、L146 `torch.topk(slash, slash_size, -1).indices - (q_len - 1)`；
- **stream_llm**：L163-164 `torch.triu(torch.tril(ones, 0), -slash_size)` 后 `mask[:, :vertical_size] = 1`，L167 再 `torch.tril`；
- **block_sparse**：L173-179 `block_num = (q_len - 1) // 32 + 1`、补零后 `.reshape(…, block_num, 32, -1).mean(-2)`；L181 块级 `matmul` 加块因果掩码（此处**没有** `/sqrt(d)`）；L183 `torch.topk(-qk, block_num - block_num//topk_ratio)`；L187-188 `repeat(1,1,1,32,1,32)` 摊回 token 网格后 `tril` 并切回 `[:q_len, :q_len]`；
- 打分 L156-157、L168-169、L190-191：`attn_weights_x = attn_weights * est_attn`，再 `.sum(-1).mean(-1)`；候选网格是 L209/L211/L213 硬编码的 `vs_list`，L218 `if score > best_score` 用严格大于，同分时自然留下更早的候选；
- L222-225 后处理：`stream_llm → vertical_and_slash`；`block_sparse → vertical_and_slash` 且 `best_v, best_s = 1000, 6096`——`pattern` 的改写就出自这里。

**本题自己的补充**（上游没写、题面显式钉死）：分数对**全部** query 行取均值，上游只统计 `attn_weights_x[:,:,2500:]` 之后的行；候选网格由入参 `candidates` 显式给出，而不是 `vs_list` 硬编码；同分一律取更小下标：候选层面还能对上 L218 的严格 `>`，但列、对角线、块三个层面的并列顺序是 `torch.topk` 根本不保证、由本题钉死的；掩码与分数逐 head 返回，上游每 head 调一次只把 `(kind, v, s, score)` 记进 config 并 `print` 一行；`block_sparse` 仍走三元 `(kind, a, b)`，其中 `b` 不参与也不校验；\(P\) 及全部中间量用 float64，上游 softmax 走 float32。

## 相关题目

- `kv-streaming-llm-sink`：StreamingLLM 的 sink + 滑窗 KV 版本；本题把同一种形状当成候选之一来评分。
- `kv-quest-page-bound`：同样"先估个便宜的分数再选"，但 Quest 每个 decode step 重选、颗粒是 page；本题搜一次就定形、颗粒是整条 pattern。
- `kv-chunk-independent-topk`：纯 top-k 记账；本题的 top-k 前面还挂着"置 \(+\infty\) 强制保留"造成的并列陷阱。
