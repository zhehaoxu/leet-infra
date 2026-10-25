# LayerSkip 早退调度题解

## 出处

LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding，Elhoushi、Shrivastava 等，FAIR at Meta，2024，[arXiv:2404.16710](https://arxiv.org/abs/2404.16710)，代码 [facebookresearch/LayerSkip](https://github.com/facebookresearch/LayerSkip)。与本题直接相关的原文段落：

- §2.1/§4.1：训练时做 layer dropout（越靠后的层 dropout 率越高）并加 early-exit loss，所有层共用同一个 LM head，使一个 checkpoint 等价于「一族不同深度的子模型的集成」。
- §4.2 早退推理：「we run the first $E$ transformer layers in a model, and skip to the model's LM head」——只跑前 $E$ 层就直接反嵌入。
- §4.3.1 Self-Drafting：用前 $E$ 层做自回归起草，起草长度即 `num_speculations`（本文的 $\gamma$）。
- §4.3.2 Self-Verification：「the self-verification stage critically only requires computing the remaining layers of the model that were not used in the draft stage. For a model with $L$ layers, the number of verification layers is $L-E$」——验证只跑剩下的 $L-E$ 层。
- §4.3.3 Reusing the Cache：单一 KV cache（草稿与验证共用前 $E$ 层的 KV）+ exit query cache（只保存退出层 $E-1$ 的 query），合称 **KVQ cache**，这正是「验证可以从第 $E$ 层接着跑」的物理前提。
- §A.4 伪代码：`verified_tokens = model.forward_remainder(input + draft_tokens, kvq_cache)`，然后 `matched_tokens = match(draft_tokens, verified_tokens)` —— 接受判据是 **draft 与全模型 argmax 是否一致**，不是拒绝采样，也不是 typical acceptance。

本题的两处「泛化/取舍」都在题面里显式写了：① 原文每个模型固定一个 $E$，本题让早退深度**逐草稿位自适应**（用置信阈值 `tau` 形式化，上限 $D$），这正是 Figure 4「多数 token 在最后一层之前若干层就已预测正确」的按 token 版本；② 原文只说验证跑 $L-E$ 层，本题 $\gamma$ 个草稿位深度不齐，一次并行前向只能复用**所有位置共同拥有**的缓存前缀 $\min_j e_j$，故代价取 $\max(L-\min_j e_j, 1)$，$1$ 是给收尾 token 的 LM head 记的账。

## 核心直觉：把「层数」变成可核算的货币

推测解码的收益来源是「便宜的算力多算几步、贵的算力少算几步」。LayerSkip 的巧妙之处在于**草稿模型和被验证的模型是同一个模型**：草稿 = 浅前缀子模型，验证 = 深前缀子模型，两者共享权重与缓存。于是每一轮的代价可以精确到「跑了几个 transformer block」：

- 草稿：$\gamma$ 次串行浅前向，第 $j$ 次跑 $e_j$ 个 block；
- 验证：一次并行前向，跑 $L-\min_j e_j$ 个 block（把所有草稿位的隐藏状态对齐到最浅的那段公共缓存，从它后面一层接着跑到第 $L$ 层）；
- 产出：一致的草稿前缀 $a$ 个 + 1 个由全模型给出的 token。

只要 $\sum_j e_j + (L - \min_j e_j) < (a+1)\,L$，这一轮就是赚的——这就是 `speedup = sum(toks)·L/sum(lays) > 1` 的直观含义。

## 为什么并行验证要对齐到 min(e_j)

若 $\gamma$ 个草稿位的早退深度分别是 $2,3,1$，那么它们各自缓存下来的「已算过的层」是 $\{1,2\},\{1,2,3\},\{1\}$。一次批量前向对所有位置执行**同一组层**：要推进到第 $L$ 层，就必须从「最浅的那个位置也还没算过」的层开始，也就是 $\min_j e_j + 1 = 2$，于是本题里 $L=5$ 时验证要跑第 $2$ 到第 $5$ 层共 $4$ 个 block。深度更大的位置会**重复计算**自己已经缓存过的层，这是批处理（同一 batch 必须同构）的固有代价，也正是「公共前缀」这一说法的来源。写成 $\max_j e_j$ 就漏掉了这份重复计算，属于实现时最容易手滑的地方。

## 代价核算与 speedup 口径

`lays` 是**逐请求**的 block 数，`speedup` 却是**整个 batch 一个数**：$\text{sum(toks)}\cdot L/\text{sum(lays)}$。它的解释是「同一批 token 若逐 token 全模型解码需要多少 block，除以实际用了多少 block」。因为分母是求和而非逐项，token 多的请求会主导结果；把它写成各请求 speedup 的平均，在有「一个请求很顺、另一个请求第一位就废」的混合 batch 里会明显偏乐观（用例 `check_speedup_is_token_weighted_not_mean_of_requests` 就是造了 2.4 与 0.667 两个极端、真值 1.4545 的例子来钉住这个口径）。

## 数值与实现坑

1. **阈值必须用二进制精确的数**。float32 里 `0.7` 实际是 0.699999988…，而 python float `0.7` 是 0.69999999999999996，两者比较会「该达标的没达标」。题面示例全部用 $1/16$ 的倍数（0.3125/0.5/0.625/0.75/0.8125），`tau=0.625` 才能让「踩线达标」这种边界既写在题面里又能在评测里复现。
2. **argmax 并列取最小下标**。`torch.argmax` 的并列行为不保证，必须自己扫（`if row[v] > m` 严格大于 + 从左到右即得最小下标）。草稿 token 与目标 token 两处都要一致，否则「一致判定」会假阳性/假阴性。
3. **验证成本一分不省**。第 0 位就不一致时本轮只产出 1 个 token，但 $\gamma$ 次草稿和一次验证照付——这是 `speedup < 1` 最常见的来源，不要「提前 break 掉层数统计」。
4. **$\max(\cdot,1)$ 的下界**。$D = L$ 且所有位置都跑满 $L$ 层时，剩余层数是 0，但收尾 token 仍要过 LM head，按 1 个 block 计；写成 $L-\min e$ 会得到 $\texttt{sum(lays)} \le 0$ 之类的荒谬值。
5. **$\gamma = 0$ 的形状**。`depth` 必须是 $(R,0)$（每请求一行空行），不是 $(0,)$；此时 `toks=1`、`lays=L`，speedup 恒为 1，可以直接当哨兵用例。
6. **早退是「第一个达标层」而不是「最浅的达标层」**：两者相同，但实现时若从 $D-1$ 反向扫（对拍写法）必须反复覆盖成更小者，否则语义反了。

## 对拍口径

`solution.py` 正向扫层、逐位累加深度、`sum(es)` 求草稿层数、`min(es)` 求验证层数、顺序扫描一致性求 $a$。`test_cases.py` 里的独立 oracle 刻意全部反着来：**反向**扫层（`for l in reversed(range(D))`）反复覆盖求最浅达标层；用 `max(range(V), key=...)`（Python 的 `max` 天然返回首个最大值）取 argmax；把「执行过的 block 编号」逐层 `list(range(1, e+1))` 具象化拼成一个清单，用 `len(blocks)` 当作层数——这条路径同时验证了代价模型本身就是「跑了几个 block」；不一致位置先物化成布尔列表再取第一个 `False`；batch 汇总用 `math.fsum` 且在 `speedup` 上按浮点口径比对。随机用例覆盖 $R,\gamma,D,V,L$ 与 4 档 `tau` 的组合，另有 8 个退化/不变量用例（$\gamma=0$ 的哨兵、`tau=0`、`tau>1`、全相同时的并列与踩线、目标全不一致、改目标分布不影响调度、抬高 `tau` 只让深度变深、$g=1$ 的最小轮）。

## 复杂度

参考解 $O(R\,\gamma\,(D + V))$ 时间、$O(R\,\gamma)$ 额外空间（`.tolist()` 展开后是纯 Python 循环，无张量分配）。oracle 因为物化 block 编号清单是 $O(R(\gamma D + L))$，仍在毫秒级——本题和 283 不同，两侧都是多项式复杂度的模拟，对拍可以放心放大到几百个随机用例。
