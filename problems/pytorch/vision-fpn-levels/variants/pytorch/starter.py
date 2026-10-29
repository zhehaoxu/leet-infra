import torch


def solve(features, lateral_weight, lateral_bias=None, include_p6=False,
          anchors_per_loc=3, strides=None):
    """FPN 自顶向下融合 + 层尺寸/参数量/anchor 账本。

    入参：
    - features: list/tuple，长度 L_in >= 1，第 i 项是 (N, C_i, H_i, W_i) 的
      backbone 特征图（按分辨率从高到低排列，例如 C2、C3、C4、C5）；各级 N 相同，
      C_i 可以各不相同。要求每项都是四维、H_i/W_i >= 1。
    - lateral_weight: list，长度必须等于 L_in，第 i 项是 (fpn_dim, C_i) 的二维
      张量，表示该级 1x1 侧接卷积核（等价于 (fpn_dim, C_i, 1, 1)）；所有级的
      fpn_dim（第 0 维）必须相同。
    - lateral_bias: 默认 None（等价于每级都无 bias）。给 list 时长度必须等于
      L_in，第 i 项是 (fpn_dim,) 的一维张量或 None（表示该级不加 bias）。
    - include_p6: bool，默认 False。True 时在最后一级（P5）之后追加 P6。
    - anchors_per_loc: int，默认 3，每个空间位置的 anchor 个数，必须 >= 1。
    - strides: 默认 None，表示第 i 级步长取 2 ** (2 + i)（即从 C2 起算的
      4/8/16/32）。给 list 时长度必须等于 L_in，每项为正整数，**按给定值原样
      记账，不由 H_i/W_i 反推**（尺寸非 2 的幂时步长与尺寸本来就不自洽；若
      features 从 C3 开始，应显式传 [8, 16, ...]）。

    计算流程（严格按此顺序）：
    1. 侧接：lateral[i] = 1x1 卷积(features[i], lateral_weight[i],
       lateral_bias[i] 若给)，输出 (N, fpn_dim, H_i, W_i)。
    2. 自顶向下：fused[L_in-1] = lateral[L_in-1]；对 i = L_in-2 .. 0，
       fused[i] = lateral[i] + upsample_nearest(fused[i+1], H_i, W_i)。
       最近邻放大到目标尺寸 (H_t, W_t) 的**整数**规则为
       src_y = (dst_y * H_up) // H_t，src_x = (dst_x * W_up) // W_t
       （向下取整、不夹取也不插值；H_t == 2*H_up 时就是每行/列复制两次）。
       因目标尺寸取自同一级的 lateral 输出，相加时尺寸天然对齐，不需要裁剪；
       若目标尺寸大于上层尺寸（不是整数倍放大），同一公式照用。
    3. P6（仅 include_p6=True）：对融合后的 P5 做 kernel_size=2、stride=2、
       padding=0 的最大池化，尺寸为 (H5 // 2, W5 // 2)（向下取整，奇数边长
       丢掉最后一行/最后一列，不做 ceil_mode 补齐）。要求 H5/W5 >= 2，否则
       抛 ValueError。P6 没有侧接卷积。

    返回 dict，键固定为：
    - "features": list，长度 L = L_in + (1 if include_p6 else 0)，依次是
      (N, fpn_dim, H, W) 的融合输出张量（P2..P5 或 P3..P5 + P6，按
      features 的顺序），dtype/device 跟随 features。
    - "sizes": int64 张量，形状 (L, 2)，每行 (H, W)。
    - "strides": int64 张量，形状 (L,)，各级步长；P6 的步长 = 2 * 最后一级
      输入步长。
    - "lateral_params": int64 张量，形状 (L_in,)，第 i 级 1x1 侧接的参数量
      = fpn_dim * C_i + (fpn_dim if 该级有 bias else 0)；**不含 P6**（P6 无
      侧接，故长度只有 L_in 项）。
    - "num_lateral_params": int，lateral_params 之和。
    - "num_upsample": int，自顶向下"上采样 + 相加"的次数 = L_in - 1
      （单级时为 0；P6 的池化不计入）。
    - "anchors_per_level": int64 张量，形状 (L,)，第 l 级 anchor 数
      = anchors_per_loc * H_l * W_l。
    - "num_anchors": int，anchors_per_level 之和。

    错误约定：features 为空或某项不是四维张量、H/W < 1；lateral_weight /
    lateral_bias / strides 长度不等于 L_in；lateral_weight[i] 不是
    (fpn_dim, C_i) 的二维张量（含 fpn_dim < 1、与 features[i] 的通道数不一致）；
    lateral_bias[i] 既不是 None 也不是 (fpn_dim,) 的一维张量；strides 每项不是
    正整数；anchors_per_loc < 1；include_p6=True 但 P5 的 H 或 W < 2——全部抛
    ValueError。不修改任何输入张量。
    """
    # TODO
    raise NotImplementedError
