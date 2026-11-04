"""BEFT (official HF PEFT tuners/beft; arXiv:2509.15974, ACL 2026) --
bypass bias fine-tuning ledger.

Dispatch by the ``mode`` keyword:

- mode="forward_delta": the Linear.forward bypass account
  (kwargs: x (..., in), weight (out, in), beft_bias (1, out) or
  (out,), base_bias (out,) or None) -> base linear output (x @ W.T
  plus the ORIGINAL bias if present) + beft_bias reshaped to (out,),
  broadcast over (..., out); the original weights/bias are untouched
- mode="adapter_sum": the multi-adapter account (kwargs: bias_list)
  -> sum of beft_bias vectors flattened to (out,); the official
  forward adds the summed bias to the base output ONCE
- mode="merge_bias": the merge state machine
  (kwargs: base_bias (out,) or None, beft_bias, safe=False) ->
  base_bias + beft_bias.squeeze(); a bias-free base layer raises
  ValueError; safe=True runs an isfinite check on the sum first and
  raises ValueError on non-finite values instead of returning them
- mode="unmerge_bias": the inverse account
  (kwargs: merged_bias, beft_bias) -> merged_bias - beft_bias,
  restoring the original bias exactly (merge/unmerge round trip)
- mode="layer_setup": the update_layer shape account
  (kwargs: out_features, init_weights=True) ->
  {"shape": [1, out_features], "param_count": out_features} plus
  "init" = zeros when init_weights (training starts at delta = 0);
  random init returns no numeric anchor

The only trainable parameter is the added (1, out) bias per value
projection; the paper's finding is that fine-tuning this b_v beats
b_q / b_k in low-data regimes.
"""

回源: HF PEFT src/peft/tuners/beft/layer.py
（论文 BEFT: Bias-Efficient Fine-Tuning of Language Models in
Low-Data Regimes, arXiv:2509.15974, ACL 2026）。机制：不改权重矩阵、
不加低秩因子——只给 value 投影**新增**一个旁路偏置 beft_bias
(1, out_features) 作为唯一可训练参数（论文结论：微调 b_v 在低数据
场景优于 b_q/b_k；对无 bias 模型即「加 bias 再训练」路径）。

forward 的确定性链（官方 Linear.forward）:
  result = base_layer(x)                    # 原权重与原 bias 原封不动
  beft_bias = sum(beft_bias[a].flatten()    # 多 adapter 先求和
                  for a in active_adapters)
  result = result + beft_bias               # (..., out) + (out,) 广播
init_weights=True 时 beft_bias 全零 —— 训练起点 delta=0。

merge/unmerge 状态机（官方 merge/unmerge）:
  merge:   base.bias += beft_bias.squeeze()   # 要求原 bias 存在，
                                              # 否则 ValueError
  safe_merge: 先 clone 相加再 torch.isfinite(...).all() 检查，
              非 finite 抛 ValueError 且不写回
  unmerge: base.bias -= beft_bias.squeeze()   # 恢复原 bias

update_layer 布局账: beft_bias = randn((1, out_features))，
init_weights=True 时 reset 为全零；每层可训练参数量 = out_features。

五入口复刻:
- forward_delta / adapter_sum / merge_bias / unmerge_bias / layer_setup
"""

import numpy as np


def forward_delta(x, weight, beft_bias, base_bias=None):
    raise NotImplementedError


def adapter_sum(bias_list):
    raise NotImplementedError


def merge_bias(base_bias, beft_bias, safe=False):
    raise NotImplementedError


def unmerge_bias(merged_bias, beft_bias):
    raise NotImplementedError


def layer_setup(out_features, init_weights=True):
    raise NotImplementedError


def solve(mode=None, **kw):
    if mode == "forward_delta":
        return forward_delta(kw["x"], kw["weight"], kw["beft_bias"],
                             base_bias=kw.get("base_bias"))
    if mode == "adapter_sum":
        return adapter_sum(kw["bias_list"])
    if mode == "merge_bias":
        return merge_bias(kw["base_bias"], kw["beft_bias"],
                          safe=kw.get("safe", False))
    if mode == "unmerge_bias":
        return unmerge_bias(kw["merged_bias"], kw["beft_bias"])
    if mode == "layer_setup":
        return layer_setup(kw["out_features"],
                           init_weights=kw.get("init_weights", True))
    raise ValueError("unknown mode: %s" % mode)
