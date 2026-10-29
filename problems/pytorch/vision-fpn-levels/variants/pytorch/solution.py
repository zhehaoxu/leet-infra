"""参考答案：FPN 自顶向下融合与层尺寸账（向量化 conv2d / index_select 实现）。

口径（与 starter docstring 逐字对应）：
- 侧接是 1x1 卷积：`lateral_weight[i]` 为 (fpn_dim, C_i) 二维张量（等价于
  (fpn_dim, C_i, 1, 1) 核），`lateral_bias[i]` 为 (fpn_dim,) 或 None；
- 侧接参数量 = fpn_dim * C_i + (fpn_dim if 有 bias else 0)；
- 自顶向下最近邻放大用整数规则 src = floor(dst * S_up / S_t)，同层尺寸对齐后
  直接相加（不需要额外裁剪），共 len(features) - 1 次上采样；
- P6（可选）由融合后的 P5 经 kernel=2 / stride=2 / padding=0 的最大池化得到，
  尺寸为 floor 除法（奇数边长丢掉最后一行/列）。

本条实现路径与判题 oracle（纯 Python 四邻域/逐像素循环）结构不同。
"""

import torch
import torch.nn.functional as F


def _nearest_resize(t, h_t, w_t):
    """最近邻放大到 (h_t, w_t)：src = floor(dst * S_up / S_t)，纯整数算术。"""
    h_u, w_u = t.shape[-2:]
    iy = (torch.arange(h_t, device=t.device) * h_u) // h_t
    ix = (torch.arange(w_t, device=t.device) * w_u) // w_t
    return t.index_select(2, iy).index_select(3, ix)


def solve(features, lateral_weight, lateral_bias=None, include_p6=False,
          anchors_per_loc=3, strides=None):
    if not isinstance(features, (list, tuple)) or len(features) == 0:
        raise ValueError("features 必须是至少含一级的非空 list/tuple")
    num_in = len(features)
    for i, f in enumerate(features):
        if not isinstance(f, torch.Tensor):
            raise ValueError(f"features[{i}] 必须是 torch.Tensor")
        if f.dim() != 4:
            raise ValueError(f"features[{i}] 必须是 (N, C, H, W) 四维，got {tuple(f.shape)}")
        if f.shape[-2] < 1 or f.shape[-1] < 1:
            raise ValueError(f"features[{i}] 的 H/W 必须 >= 1，got {tuple(f.shape)}")

    if len(lateral_weight) != num_in:
        raise ValueError(
            f"lateral_weight 长度必须等于特征级数 {num_in}，got {len(lateral_weight)}")
    for i, w in enumerate(lateral_weight):
        if not isinstance(w, torch.Tensor) or w.dim() != 2:
            raise ValueError(f"lateral_weight[{i}] 必须是二维张量")
    if lateral_bias is None:
        lateral_bias = [None] * num_in
    elif len(lateral_bias) != num_in:
        raise ValueError(
            f"lateral_bias 长度必须等于特征级数 {num_in}，got {len(lateral_bias)}")

    fpn_dim = int(lateral_weight[0].shape[0])
    if fpn_dim < 1:
        raise ValueError(f"fpn_dim 必须 >= 1，got {fpn_dim}")
    if anchors_per_loc < 1:
        raise ValueError(f"anchors_per_loc 必须 >= 1，got {anchors_per_loc}")

    lateral = []
    lateral_params = []
    for i, (f, w, b) in enumerate(zip(features, lateral_weight, lateral_bias)):
        if w.dim() != 2:
            raise ValueError(
                f"lateral_weight[{i}] 必须是 (fpn_dim, C_in) 二维，got {tuple(w.shape)}")
        if w.shape[0] != fpn_dim or w.shape[1] != f.shape[1]:
            raise ValueError(
                f"lateral_weight[{i}] 形状应为 ({fpn_dim}, {f.shape[1]})，"
                f"got {tuple(w.shape)}")
        if b is not None:
            if not isinstance(b, torch.Tensor) or b.dim() != 1 or b.shape[0] != fpn_dim:
                got = tuple(b.shape) if isinstance(b, torch.Tensor) else type(b).__name__
                raise ValueError(f"lateral_bias[{i}] 形状应为 ({fpn_dim},)，got {got}")
        lateral_params.append(int(fpn_dim * f.shape[1] + (fpn_dim if b is not None else 0)))
        kernel = w[:, :, None, None]
        lateral.append(F.conv2d(f, kernel, b))

    # 自顶向下：上层输出放大到下一级尺寸后相加（尺寸已对齐，无需裁剪）
    fused = [None] * num_in
    fused[-1] = lateral[-1]
    for i in range(num_in - 2, -1, -1):
        h_t, w_t = lateral[i].shape[-2:]
        fused[i] = lateral[i] + _nearest_resize(fused[i + 1], h_t, w_t)
    num_upsample = num_in - 1

    if include_p6:
        p5 = fused[-1]
        if p5.shape[-2] < 2 or p5.shape[-1] < 2:
            raise ValueError(
                f"include_p6=True 要求 P5 的 H/W >= 2，got {tuple(p5.shape[-2:])}")
        fused.append(F.max_pool2d(p5, kernel_size=2, stride=2))

    if strides is None:
        base = [2 ** (2 + i) for i in range(num_in)]
    else:
        if len(strides) != num_in:
            raise ValueError(f"strides 长度必须等于特征级数 {num_in}，got {len(strides)}")
        base = [int(s) for s in strides]
        if any(s < 1 for s in base):
            raise ValueError(f"strides 每项必须 >= 1，got {base}")
    full_strides = base + ([2 * base[-1]] if include_p6 else [])

    sizes = [[int(t.shape[-2]), int(t.shape[-1])] for t in fused]
    anchors = [int(anchors_per_loc) * h * w for h, w in sizes]
    return {
        "features": fused,
        "sizes": torch.tensor(sizes, dtype=torch.int64),
        "strides": torch.tensor(full_strides, dtype=torch.int64),
        "lateral_params": torch.tensor(lateral_params, dtype=torch.int64),
        "num_lateral_params": int(sum(lateral_params)),
        "num_upsample": int(num_upsample),
        "anchors_per_level": torch.tensor(anchors, dtype=torch.int64),
        "num_anchors": int(sum(anchors)),
    }
