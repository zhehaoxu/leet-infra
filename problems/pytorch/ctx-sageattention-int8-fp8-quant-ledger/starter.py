CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


# 本题是 SageAttention 量化路径的三段纯数值账：per-block INT8 的两套舍入口径、
# per-channel FP8 的统计推导、以及转置-填充-置换的布局。三段都不做 attention
# 前向、不做 kernel 调度。
#
# mode 列表：
#   int8_block_quant     —— 入参 x（token × head_dim）/ blk / sm_scale /
#                           rounding（"rne" 或 "away"）/ amax_floor
#                           返回 amax / scale / scaled / x_int8 /
#                           num_blocks / head_dim
#   fp8_channel_scale    —— 入参 x（token × head_dim）/ scale_max / smooth_v
#                           返回 mean / amax / scale / recp_scale /
#                           padded_tokens / head_dim
#   transpose_pad_permute—— 入参 x（token × head_dim）/ pad_size
#                           返回 padded_tokens / padded / permuted / permute_index
#
# 所有 mode 都返回 dict，字段含义见 description.md。
def solve(mode: str, **kw):
    raise NotImplementedError
