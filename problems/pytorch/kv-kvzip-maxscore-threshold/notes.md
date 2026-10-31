# 解题说明 · KVzip 的聚合与全局阈值

## 一句话

把 prefill 期间所有 query 的注意力对每个 KV pair 取 **max**，再跨层跨头排一次序，用**下标** `int(T*ratio)-1` 处的分数当阈值，严格大于它的才留下。

## 为什么是 max

`_get_score` 的关键三行是：query 摊成 `(bsz, n_kv, group, q_len, d)` 后算 `softmax(qk/√d + causal)`，切掉 sink 前缀，然后 `attn.amax(dim=(-3, -2))`。一个 KV pair 只要被**任意一个** query 位置、**任意一个**同组 query 头强烈关注过，它就可能影响输出，所以"最大"才是保守估计：mean 会抹平稀疏尖峰，sum 会被 group 大小与 query 数量缩放。max 可交换，"先组内再 query"与一次 `amax` 等价。

## 三处 off-by-one

1. `n = max(int(T*ratio)-1, 0)`：`-1` 让阈值正好落在"要保留的最后一个"上；
2. `valid = score > thres` 是**严格**大于：阈值元素自己被丢，真实保留数最多是 n，永远到不了 ratio 那么多；
3. `max(...,0)`：`T*ratio < 1` 时下标是 -1，上游夹到 0，于是阈值 = 最大值、结果为**空集**。这是"至少留一条"的反面，想兜底得自己加。

`keep_ratio` 是 `valid.float().mean()`（上游 `prune` 的 `r_ = 1 - rmv.mean().item()`），它常常明显小于 `ratio`。

## 两种 level

- `pair`：把各层分数 stack 起来整块 flatten，跨层跨头共用一个阈值，强头会吃掉名额；
- `pair-uniform`：`k = int(N*ratio)`，每层每头各自 topk，只看名次不看数值，且上游这条路径直接 `return valids, 0`，所以阈值恒为 0；
- `--level head` 是另一条路径（上下文无关的头级分数），不在此题。

## 判分点

- `scores` 必须是 group×query 上的 max（单点尖峰要被抓到，mean 过不了）；
- `threshold_index` 是下标：T=20、ratio=0.3 时是 5 而不是 6；
- 严格大于：全同分时阈值就是那个值，结果为全 False；
- `ratio >= 1` 与 `pair-uniform` 两支的 `threshold`、`threshold_index` 都是 0；
- 同分取更小的 key 下标（上游 `torch.topk` 不定义同分顺序）。

## 边界与复杂度

- 上游打分路径里没有 value-norm（grep `linalg.norm` 无命中），不要自己加一项。
- 聚合 \(O(L H G Q N)\)，排序 \(O(T\log T)\)；这是 prefill 后的一次性开销，不进 decode 热路径。
- 内部统一 float64：float32 下阈值附近的并列会因为舍入抖动，而"严格大于"对并列极其敏感。

## 出处

- `attention/score.py` L44 `query_states.view(bsz, num_kv, -1, q_len, head_dim)`；L61 softmax 注释 `# not fp32`；L62 `attn_weights[..., self.sink:self.sink + ctx_len]`；L63 `score = attn_weights.amax(dim=(-3, -2))  # max over group, q`。
- `attention/score.py` L88-102 `_threshold`：L94 `torch.sort(score.reshape(-1), descending=True)`、L95 `n = max(int(len(score_sort) * ratio) - 1, 0)`、L96 `thres = score_sort[n].item()`、L97 `torch.where(score > thres, True, False)`、L98-100 `ratio >= 1` 全留且 `thres = 0.`。
- `attention/score.py` L104-120 `_threshold_uniform`：L111 `k = int(n_seq * ratio)`、L112 `torch.topk(score, k, dim=-1)`、L120 `return valids, 0`。
- `attention/kvcache.py` L123-139 `prune`：L126 `if "uniform" in level`、L133 `r_ = 1 - rmv.mean().item()`；L168 `lens_k_head = valid.sum(-1)  # length of retained KV per head`。
- `args.py` L4 `-r/--ratio default=0.3`、L12-16 `--level` 的 `pair/head/pair-uniform`。

## 与 kv-adakv-headwise 的区别

AdaKV 也是对 flatten 后的分数做全局 top-k，但它的分数是上游算好的，exam point 是"剪多少 + 头保底"；本题的 exam point 是**分数怎么聚合**（group×query 的 max）与**阈值下标算术**，并多一条 `pair-uniform` 的 per-head 分支。

## 相关题目

- `kv-adakv-headwise`：跨头 flatten 全局 top-k + 头保底。
- `kv-chunk-independent-topk`：对已知分数做分块 topk。
- `kv-h2o-heavy-hitters`：按累计注意力逐 token 驱逐，颗粒度不同。
