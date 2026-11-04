"""BEFT 的旁路 bias 适配账。

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
    """BEFT forward：base 线性层输出 + 旁路 bias（广播）。

    x: (..., in)；weight: (out, in)；beft_bias: (1, out) 或 (out,)；
    base_bias: (out,) 或 None（无 bias 层）。返回 (..., out)。
    """
    out = x @ np.asarray(weight).T
    if base_bias is not None:
        out = out + np.asarray(base_bias)
    return out + np.asarray(beft_bias).reshape(-1)


def adapter_sum(bias_list):
    """多 active adapter 的 beft_bias 求和账（官方先 flatten 累加再一次加到 result）。"""
    total = None
    for b in bias_list:
        cur = np.asarray(b).reshape(-1)
        total = cur if total is None else total + cur
    return total


def merge_bias(base_bias, beft_bias, safe=False):
    """merge：base.bias += beft_bias.squeeze()。

    base bias 不存在（None）-> ValueError；safe=True 先算和再做
    finite 检查，非 finite 抛 ValueError（不返回损坏结果）。
    """
    if base_bias is None:
        raise ValueError("Base layer has no bias, cannot merge bias adapter")
    bb = np.asarray(beft_bias).reshape(-1)
    merged = np.asarray(base_bias) + bb
    if safe and not np.isfinite(merged).all():
        raise ValueError(
            "NaNs detected in the merged weights. The adapter seems to be broken")
    return merged


def unmerge_bias(merged_bias, beft_bias):
    """unmerge：base.bias -= beft_bias.squeeze()（恢复原 bias）。"""
    return np.asarray(merged_bias) - np.asarray(beft_bias).reshape(-1)


def layer_setup(out_features, init_weights=True):
    """update_layer 布局账：beft_bias 形状 (1, out)。

    init_weights=True -> 全零（训练起点 delta=0）；False -> 随机
    （官方 randn，不出确定性锚，只回 shape 与 init 标志）。
    """
    setup = {"shape": [1, out_features], "param_count": out_features}
    if init_weights:
        setup["init"] = np.zeros((1, out_features))
    return setup


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
