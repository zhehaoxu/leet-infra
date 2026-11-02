import torch


def solve(values, block_size, block_scales, global_scale=1.0):
    # TODO: 把 values 按 block_size 分块，量化到 FP4 E2M1（1 符号 + 2 指数 + 1 尾数，bias=1）。
    #   - 正半轴网格：0, 0.5, 1, 1.5, 2, 3, 4, 6；0.5 是唯一的非零次正规（没有 0.25 这一档）。
    #   - 编码：先算 eff = global_scale * block_scales[i // block_size]，再算 scaled = v / eff，
    #     然后按「最近邻 + 平局取尾数位 0（偶）」选码；|scaled| > 6 饱和到 ±6 并记 sat。
    #   - 码值：bit3 = 符号（含 -0.0，符号必须保留），bit2..1 = 指数，bit0 = 尾数。
    #   - 返回 {"codes", "dequant", "eff_scale", "sat_mask", "tie_mask", "block_sat", "block_ties", "grid"}。
    #   - 校验：block_size 为正整数（bool 不算）、block_scales 长度 = ceil(n / block_size)、
    #     每项有限且 > 0、global_scale 有限且 > 0、values 不得含 NaN；空输入合法。
    raise NotImplementedError
