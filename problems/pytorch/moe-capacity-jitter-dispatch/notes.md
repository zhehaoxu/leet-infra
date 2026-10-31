# 参考做法：把优先级排队写成一次显式贪心

## 实现

1. 候选集合用 `torch.sort(probs, dim=-1, descending=True, stable=True)` 的前 $k$ 列。`stable=True` 让概率并列时保留「专家索引升序」，这就是题面说的并列口径；写成 `torch.topk` 在 CPU 小尺寸上通常给同样的结果，但并列顺序不是契约的一部分，题面只约束结果。
2. `key = sorted_probs[:, :k] + jitter.gather(1, cand)`，`order = torch.argsort(-key.reshape(-1), stable=True)`。取负键 + 稳定排序 = 先按 `key` 降序、再按槽位编号升序，与题面 `(-key, s)` 完全一致。
3. 占位写成 Python 循环：`remaining` 是每专家的余量列表，`used` 是每 token 已占用专家的集合，`for slot in order.tolist(): token, rank = divmod(slot, k)`，然后沿 `cand[token][rank:]` 找第一个「有余量且未被本 token 占用」的专家。
4. 权重：`kept = assign >= 0`，`kept_probs = torch.where(kept, probs.gather(1, assign.clamp(min=0)), 0.0)`，`denom = kept_probs.sum(1, keepdim=True)`，`weights = torch.where(denom > 0, kept_probs / denom, 0.0)`。
5. `counts = torch.bincount(assign[kept], minlength=E)[:E]`，`dropped = T*k - int(kept.sum())`。

四个常见错误：

- **让 jitter 参与 top-k**。`topk(probs + jitter)` 会换掉候选集合，那是「显式噪声门控」的口径，与本题「jitter 只决定排队顺序」不同。
- **按 token 顺序（槽位升序）处理**。形状照样对，`assign` 却完全不同：容量分派的语义就藏在处理顺序里。
- **`assign` 里的 `-1` 直接进 `gather`**。要么 `RuntimeError`，要么先 `clamp(min=0)` 却忘了用 `where` 置零，丢弃位置于是拿到专家 0 的概率，把归一化分母算大。
- **忘了「同一 token 不重复占同一专家」**。低秩槽位先被处理时，高秩槽位可能已经把某个专家占掉；不加 `used` 判定就会两个槽位挤同一专家，`counts` 也随之失真。

## 正确性检验

8 个 `check_*`：手算示例（五个返回项全部核对）、随机对拍（8 组 `T/e/k/capacity/jitter_amp` 组合 + 三种非连续视图）、分派不变量（`order` 是排列且沿方向键值非升、`counts <= capacity`、`counts.sum() == Tk - dropped`、同 token 存活专家互异、被分派专家必须来自候选集且秩不早于自己、存活权重行和为 1 或整行为 0）、无竞争与全丢弃（`capacity=100` 时必须退化成朴素 top-k+renorm，`capacity=0` 时全 `-1`）、抖动改序与溢出（同一组概率，加不加抖动让幸存者从 rank0 换到 rank1；再用负抖动把键压平，验证并列时按槽位编号）、退化与并列（概率全相等、整行为 0、单槽位、`k=E`，以及两个 64 槽位全整数键的重并列用例）、非法输入、最大形状。

oracle 是纯 Python：`sorted(range(E), key=lambda e: (-p[t][e], e))[:k]` 选候选，`sorted(range(T*k), key=lambda s: (-keys[s//k][s%k], s))` 定顺序，用列表做 `remaining`/`used` 记账，归一化用 `math.fsum`。完全不使用 torch 的 `sort`/`argsort`/`topk`/`gather`/`bincount`。

变异测试抓到：jitter 进 top-k、按槽位顺序处理、`argsort` 不加 stable、溢出链回绕到秩 0、去掉「本 token 已占用」判定、丢弃位置不置零、按候选集而不是存活集归一化、`counts` 统计候选而非分派结果、分母不加保护导致 NaN、把「秩变了」当成「被丢弃」。其中 `argsort` 不加 `stable` 只在 64 槽位的重并列用例上暴露，所以那个用例是钉住并列口径的必要用例。E ≤ 8 时行内 `torch.sort` 不加 stable 在本机与稳定排序不可区分——题面写死并列口径是为了让实现者知道自己依赖了它，而不是因为判题能测出每一种违反。

## 范围与来源

**有手有据的部分**：优先级式容量分派与「丢弃之后再归一化」都能在 `transformers/models/nllb_moe/modeling_nllb_moe.py` 的 `NllbMoeTop2Router.route_tokens` 里读到（NLLB-MoE 是 GShard 思路的官方实现）。相关行（配置默认 `batch_prioritized_routing=False`、`normalize_router_prob_before_dropping=False`）：

```text
# sort tokens based on their routing probability
# to make sure important tokens are routed, first
importance_scores = -1 * router_probs.max(dim=1)[0]
sorted_top_1_mask = top_1_mask[importance_scores.argsort(dim=0)]
sorted_cumsum1 = (torch.cumsum(sorted_top_1_mask, dim=0) - 1) * sorted_top_1_mask
locations1 = sorted_cumsum1[importance_scores.argsort(dim=0).argsort(dim=0)]
locations2 = sorted_cumsum2[...]
locations2 += torch.sum(top_1_mask, dim=0, keepdim=True)
top_1_mask = top_1_mask * torch.lt(locations1, self.expert_capacity)
top_2_mask = top_2_mask * torch.lt(locations2, self.expert_capacity)
top_1_max_probs = (router_probs * top_1_mask).sum(dim=1)
denom_s = torch.clamp(top_1_max_probs + top_2_max_probs, min=torch.finfo(router_probs.dtype).eps)
```

对应关系：上游按「路由概率降序」排队再算位置，与本题的「按 `key` 降序贪心」同构；`locations2 += sum(top_1_mask)` 把次选专家的位置往后推，就是「第一志愿满了要插到已有人数之后」的溢出数学；`top_1_mask/top_2_mask` 先被容量过滤、再拿去算归一化分母，就是本题「存活者优先归一化」；`clamp(min=eps)` 则是本题「分母为 0 给 0」的另一种写法（本题选择显式判定而不是引入一个任意小的 eps）。

**没有取到一手来源的部分**：给分数加均匀扰动再分派（通常称作 jitter）在社区里流传很广，但本次未能核对原文的具体公式，所以「`key = probs + jitter`、只沿秩向前溢出、同一 token 不重复占位」这三条属于**题面自定**口径，不做归属声明。抖动做成输入张量而不是内部 RNG，是为了消掉随机性、让判题可复现。

与合集内其他题的分工：`moe-capacity-dispatch` 是「按 token 位置顺序丢弃」的极简口径，本题是「按抖动优先级排队 + 溢出转投」；`moe-noisy-topk-gating` 的噪声**参与** top-k 选择，本题的 jitter 不参与（上游 `second_expert_policy="sampling"` 的 Gumbel 噪声也是加在选择之前）；候选、权重与专家的索引代数见 `moe-grouped-execution-plan`；跨步骤的负载账本见 `moe-dead-expert-ledger`。
