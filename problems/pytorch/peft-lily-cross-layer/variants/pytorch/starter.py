"""Lily (official yibozhong/lily, ACL 2025 Findings; arXiv:2407.09946) --
cross-layer interconnected adaptation ledger.

Dispatch by the ``mode`` keyword:

- mode="router_probs": the data-dependent router account
  (kwargs: hidden (…, hidden_dim), router_weight (ne, hidden_dim))
  -> softmax over the expert axis of hidden @ router_weight.T,
  shape (…, ne); the router is a bias-free Linear
- mode="expert_mix": the expert combination account
  (kwargs: probs_mean (ne,), hps (ne, hidden, out)) ->
  einsum("e,eio->io", probs_mean, hps), the (hidden, out) combined HP
- mode="forward_delta": the full lily_adapter.forward chain
  (kwargs: x (B,N,in), lp_weight (hidden,in), router_weight, hps,
  mlp=False) -> hidden = lp(x) [ReLU if mlp else Identity],
  probs = router(hidden), probs_mean = probs.mean(axis=(0,1)) -- the
  GLOBAL mean over batch and sequence --, combined = expert_mix,
  delta = hidden @ combined
- mode="monoscale_delta": the router-free uniform variant
  (kwargs: x, lp_weight, hps, mlp) -> combined = hps.sum(0) * (1/ne)
- mode="lp_groups": the set_lily LP-placement account
  (kwargs: n_layers, ne) -> {"stride" = n_layers // ne,
  "lp_group" = per-layer LP id (i // stride), "num_lp"};
  a new LP is placed wherever idx % stride == 0, so layers in the
  same group share one LP; HPs are model-level zero-initialized
  experts shared by ALL LPs

Dropout is fixed at 0; LP/router weights and HPs are all explicit
inputs, and only the deterministic interconnection accounts are
replicated.
"""

回源: yibozhong/lily natural-language-understanding/lily.py
（论文 Low-Rank Interconnected Adaptation across Layers, arXiv:2407.09946,
ACL 2025 Findings）。机制：拆开 LoRA 的逐层 A-B 配对——低维投影器 LP
留在层组局部（每 stride 层换新），高维投影器 HP 从各层剥离为**模型级
共享 expert 池**（ne 个 (hidden, out) 矩阵，零初始化），每个 LP 连接
全部 HP。权重更新不再被单一固定投影矩阵钉死在低秩子空间里——层间
互联打破 LoRA 的低秩约束。

lily_adapter.forward 的确定性链:
  hidden = non_linear(lp(x))                    # MLP 位 ReLU、q/v 位 Identity
  logits  = router(hidden)                      # nn.Linear(hidden, ne, bias=False)
  probs   = softmax(logits, dim=-1)             # [B, N, ne]
  p_mean  = probs.mean(dim=(0, 1))              # token 维全局平均 -> (ne,)
  combined_hp = einsum("e,eio->io", p_mean, hps)  # expert 加权组合
  delta   = hidden @ combined_hp
数据依赖 router（无 top-k、无噪声，纯 dense softmax 加权），锚点全确定。

monoscale 变体: combined_hp = sum(hps, 0) * (1/ne)（均匀平均，无 router）。

set_lily 集成账: stride = n_layers // ne，idx % stride == 0 处放新 LP
（层组共享）；HP `torch.zeros(ne, dim, out)` 零初始化 => 训练起点
delta = 0；attention 侧 delta 乘全局 scale s。

五入口复刻（LP/router 权重全部显式入参）:
- expert_mix / router_probs / forward_delta / monoscale_delta / lp_groups
"""

import numpy as np


def router_probs(hidden, router_weight):
    raise NotImplementedError


def expert_mix(probs_mean, hps):
    raise NotImplementedError


def forward_delta(x, lp_weight, router_weight, hps, mlp=False):
    raise NotImplementedError


def monoscale_delta(x, lp_weight, hps, mlp=False):
    raise NotImplementedError


def lp_groups(n_layers, ne):
    raise NotImplementedError


def solve(mode=None, **kw):
    if mode == "router_probs":
        return router_probs(kw["hidden"], kw["router_weight"])
    if mode == "expert_mix":
        return expert_mix(kw["probs_mean"], kw["hps"])
    if mode == "forward_delta":
        return forward_delta(kw["x"], kw["lp_weight"], kw["router_weight"],
                             kw["hps"], mlp=kw.get("mlp", False))
    if mode == "monoscale_delta":
        return monoscale_delta(kw["x"], kw["lp_weight"], kw["hps"],
                               mlp=kw.get("mlp", False))
    if mode == "lp_groups":
        return lp_groups(kw["n_layers"], kw["ne"])
    raise ValueError("unknown mode: %s" % mode)
