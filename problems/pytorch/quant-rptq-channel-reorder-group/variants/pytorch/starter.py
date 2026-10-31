import torch


def solve(xmax, xmin, W, num_groups, bits):
    """RPTQ：按通道量化跨度排序重排，再逐组做组内非对称量化。

    `xmax/xmin` 是形状 [C] 的逐通道激活最大值/最小值；`W` 是 [O, C] 权重。
    1. 跨度 span_i = xmax_i - xmin_i；`perm = 按 (span, 通道号) 升序稳定排序`。
    2. 把 perm 顺序的通道**均衡**切成 g 段：前 C%g 段各多 1 个；
       `counts[j]` 为第 j 段长度。
    3. 每段连续：组 xmax/xmin 取段内 max/min；
       scale = clamp((组max-组min) * 2^-bits, 1e-4, 1e4)；
       zero  = round(clamp((组max+组min) * -0.5/scale, -1e4, 1e4))。
    4. 逐元素 fake-quant：q=clamp(round(w/scale)+zero, -2^(bits-1), 2^(bits-1)-1)，
       重建 (q-zero)*scale，再按 `perm` 放回原通道顺序。

    返回 dict：perm/counts/zero_points int64，scales/dequant float32。
    非法输入（类型、范围、形状、非有限值、xmin > xmax）抛 ValueError。
    """
    # TODO
    raise NotImplementedError
