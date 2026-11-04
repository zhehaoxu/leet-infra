"""MiSS (official Joluck/MiSS, now in HF PEFT tuners/miss; arXiv:2409.15371,
ICLR 2026) -- matrix shard sharing ledger.

Dispatch by the ``mode`` keyword:

- mode="split_dims": the update_layer validation/layout account
  (kwargs: in_features, out_features, r, init_weights="miss",
  mini_r=1) -> {"n_blocks", "pad_size", "miss_block", "zero_init"};
  r <= 0 or r > in_features raises ValueError; "bat" requires
  in/out divisible by r; "mini" requires out divisible by mini_r;
  the miss_block is ZERO-initialized (delta = 0 at the start)
- mode="shard_sum": the input aggregation account
  (kwargs: x (…, in), r) -> right zero-pad to n_blocks*r then
  reshape (…, n_blocks, r) and sum across shards -> (…, r)
- mode="forward_delta": the non-bat forward delta
  (kwargs: x, miss_block, r, init_weights, mini_r, out_features);
  "mini" first repeats (r, mini_r) to (r, out) along the output axis,
  then shard_sum(x) @ miss_block
- mode="expand_delta": the merge dW account -- every input shard
  position carries the full miss_block^T, i.e.
  dW = tile(miss_block.T, (1, in//r)) shape (out, in);
  mathematically dW x == shard_sum(x) @ miss_block
- mode="bat_delta": the bat block-level delta account
  (kwargs: w0 (out, in), miss_block (out//r, r, r), r) -> dW with
  dW[j-th out block, i-th in block] = W0_block @ B[j] + B[j]

Dropout is fixed at 0 and every model call is an explicit input; only
the deterministic shard-sharing accounts are replicated.
"""

回源: HF PEFT src/peft/tuners/miss/layer.py（MissLinear.forward /
get_delta_weight_miss / bat delta 账，2024.11 已入 PEFT 官方；论文
arXiv:2409.15371，ICLR 2026，官方仓 Joluck/MiSS 的 flashop/miss.py
line 56-57 给出同语义 PyTorch 参考实现）。论文机制：抛弃 LoRA 的
双矩阵 B·A，只留一个**零初始化**小矩阵 miss_block，用"分片共享"
复制机制把它扩展成低秩更新 ΔW——单矩阵优化、参数量 O(d·r)。

三种 init_weights 形态的确定性账:
- "miss"（默认 expand）: miss_block (r, out)。前向把输入按 r 分片、
  跨片求和聚合成 r 维（in 不整除则右侧零 pad），再乘 miss_block；
  merge 时 ΔW 的每个输入分片位置都是完整的 miss_blockᵀ——
  即 ΔW = miss_blockᵀ 沿输入维 tile in//r 次，两者数学等价
  （ΔW·x = miss_block·Σ块x）。
- "mini": miss_block (r, mini_r)，先沿输出维 repeat out//mini_r 次
  扩展成 (r, out)，之后与 miss 同账。
- "bat": miss_block (out//r, r, r)，基座权重按 (out//r, r, in//r, r)
  分块，第 j 输出块走 W块 @ miss_block[j] + miss_block[j]（加性项
  破除 Bone 的共线更新）；merge 直接改权重，前向走 F.linear。

五入口复刻（dropout=0 的 Identity 分支不设参）:
- split_dims: update_layer 的校验与布局账（r>0、r<=in；bat 要求
  in/out 均被 r 整除、mini 要求 out 被 mini_r 整除）+ pad 账
- shard_sum: 输入聚合账（pad + reshape + 跨片求和 -> (…, r)）
- forward_delta: 非 bat 前向 delta（mini 先 repeat，再 shard_sum @ B）
- expand_delta: merge 的 ΔW 展开账（tile miss_blockᵀ）
- bat_delta: bat 的块级 delta 权重账
"""

import torch


def split_dims(in_features, out_features, r, init_weights="miss", mini_r=1):
    raise NotImplementedError


def shard_sum(x, r):
    raise NotImplementedError


def forward_delta(x, miss_block, r, init_weights="miss", mini_r=1,
                  out_features=None):
    raise NotImplementedError


def expand_delta(miss_block, in_features, r, init_weights="miss", mini_r=1,
                 out_features=None):
    raise NotImplementedError


def bat_delta(w0, miss_block, r):
    raise NotImplementedError


def solve(mode=None, **kw):
    if mode == "split_dims":
        return split_dims(kw["in_features"], kw["out_features"], kw["r"],
                          init_weights=kw.get("init_weights", "miss"),
                          mini_r=kw.get("mini_r", 1))
    if mode == "shard_sum":
        return shard_sum(kw["x"], kw["r"])
    if mode == "forward_delta":
        return forward_delta(kw["x"], kw["miss_block"], kw["r"],
                             init_weights=kw.get("init_weights", "miss"),
                             mini_r=kw.get("mini_r", 1),
                             out_features=kw.get("out_features"))
    if mode == "expand_delta":
        return expand_delta(kw["miss_block"], kw["in_features"], kw["r"],
                            init_weights=kw.get("init_weights", "miss"),
                            mini_r=kw.get("mini_r", 1),
                            out_features=kw.get("out_features"))
    if mode == "bat_delta":
        return bat_delta(kw["w0"], kw["miss_block"], kw["r"])
    raise ValueError("unknown mode: %s" % mode)
