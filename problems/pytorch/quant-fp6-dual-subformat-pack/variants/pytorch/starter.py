import torch


def solve(values, fmt="auto"):
    # TODO: 把 values 量化到 FP6 的两种子格式之一，并做 4-in-3 字节打包。
    #   - e3m2（1-3-2, bias=3）：max 28，最小次正规 0.0625，动态范围大；
    #     e2m3（1-2-3, bias=1）：max 7.5，最小次正规 0.125，精度高。
    #   - auto：amax <= 7.5 用 e2m3，amax <= 28 用 e3m2，超过 28 抛 ValueError。
    #   - 编码：最近邻；平局取尾数位 0（偶）；
    #     |x| 超过该格式 max 时饱和到 max 码点并置 sat_mask。
    #   - 打包：每 4 个 6-bit 码装 3 字节（小端位流）：
    #       b0 = c0 | (c1 & 3) << 6
    #       b1 = (c1 >> 2) | (c2 & 0xF) << 4
    #       b2 = (c2 >> 4) | c3 << 2
    #     不足 4 个补 0 码，pad_count 如实回报；unpacked 由 packed 反解得到。
    #   - 返回 {"fmt", "codes", "unpacked", "packed", "dequant", "sat_mask", "tie_mask", "pad_count"}。
    raise NotImplementedError
