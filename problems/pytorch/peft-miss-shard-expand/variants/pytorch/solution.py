"""MiSS 的分片共享账（Matrix Shard Sharing）。

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
    """update_layer 的校验与布局账。"""
    if r <= 0:
        raise ValueError(
            "`r` should be a positive integer value but the value passed is %d" % r)
    if r > in_features:
        raise ValueError(
            "`r` (%d) must be less than or equal to in_features (%d)" % (r, in_features))
    dims = {
        "n_blocks": in_features // r,           # 输入聚合片数
        "pad_size": (r - in_features % r) % r,  # 右侧零 pad
    }
    if init_weights == "bat":
        if in_features % r != 0 or out_features % r != 0:
            raise ValueError("The weight matrix must be fully divisible into [r, r] blocks.")
        dims["miss_block"] = (out_features // r, r, r)
    elif init_weights == "mini":
        if out_features % mini_r != 0:
            raise ValueError(
                "mini_r is divided along the out_features dimension. For optimal performance and implementation simplicity, "
                "it is recommended that out_features be divisible by mini_r. "
                "Error: %d %% %d != 0" % (out_features, mini_r))
        dims["miss_block"] = (r, mini_r)
    else:
        dims["miss_block"] = (r, out_features)
    dims["zero_init"] = True  # 零初始化 => 训练起点 ΔW = 0
    return dims


def _mini_expand(miss_block, out_features, mini_r):
    """mini 形态的 repeat 扩展: (r, mini_r) -> (r, out)。"""
    return miss_block.repeat(1, out_features // mini_r)


def shard_sum(x, r):
    """输入聚合账：右侧零 pad 到 n_blocks*r，按片内偏移对齐跨片求和 -> (…, r)。"""
    in_features = x.shape[-1]
    if in_features % r != 0:
        pad = (r - in_features % r) % r
        x = torch.nn.functional.pad(x, (0, pad))
    nb = x.shape[-1] // r
    return x.reshape(*x.shape[:-1], nb, r).sum(dim=-2)


def forward_delta(x, miss_block, r, init_weights="miss", mini_r=1,
                  out_features=None):
    """非 bat 前向 delta：mini 先沿输出维 repeat，再 shard_sum @ B。"""
    block = miss_block
    if init_weights == "mini":
        if out_features is None:
            raise ValueError("mini 形态需要提供 out_features 以确定 repeat 次数")
        block = _mini_expand(block, out_features, mini_r)
    return shard_sum(x, r) @ block


def expand_delta(miss_block, in_features, r, init_weights="miss", mini_r=1,
                 out_features=None):
    """merge 的 ΔW 展开账：每个输入分片位置放完整的 miss_blockᵀ。

    ΔW (out, in)：dW[:, i*r+j] = miss_block[j, :] 对每个块 i，
    等价于 ΔW = miss_blockᵀ 沿输入维 tile in//r 次。
    """
    block = miss_block
    if init_weights == "mini":
        if out_features is None:
            raise ValueError("mini 形态需要提供 out_features 以确定 repeat 次数")
        block = _mini_expand(block, out_features, mini_r)
    nb = in_features // block.shape[0]
    return torch.tile(block.T, (1, nb))


def bat_delta(w0, miss_block, r):
    """bat 的块级 delta 权重账：ΔW[j输出块, i输入块] = W0块 @ B[j] + B[j]。

    w0: (out, in)；miss_block: (out//r, r, r)；
    返回 ΔW (out, in)——merge 语义下基座权重被原位加上它。
    """
    out_features, in_features = w0.shape
    nb_in = in_features // r
    nb_out = out_features // r
    wb = (w0.reshape(nb_out, r, nb_in, r)
          .permute(2, 0, 1, 3))  # (nb_in, nb_out, r, r)
    res = torch.empty_like(wb)
    for j in range(nb_out):
        res[:, j] = wb[:, j] @ miss_block[j] + miss_block[j]
    return res.permute(1, 2, 0, 3).reshape(out_features, in_features)


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
