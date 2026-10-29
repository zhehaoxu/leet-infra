import torch


# 参数契约 / Parameter contract
# pos_embed: CPU float32/float64 [1,P+H0*W0,D]，1<=D<=16，可非连续，不得修改。
#            CPU float32/float64 tensor [1,P+H0*W0,D], 1<=D<=16; may be noncontiguous.
# 全部数值有限，空间 token 绝对值<=10**4；prefix 可为任意有限值。
# All values finite; spatial tokens have absolute value <=10**4, prefixes any finite value.
# old_hw/new_hw: Python 整数对 (H0,W0)/(H1,W1)，每个尺寸 1..16，允许矩形。
#                Python integer pairs (H0,W0)/(H1,W1), each size in 1..16; rectangles allowed.
# num_prefix_tokens: Python 整数 P，0<=P<=4，默认 1；前 P 项不属于空间网格。
#                    Python integer P in 0..4, default 1; prefix tokens are not spatial.
# 若 pos_embed.shape[1] != P+H0*W0，抛 ValueError（须在恒等返回前检查）。
# Raise ValueError if pos_embed.shape[1] != P+H0*W0, even for equal old/new sizes.
# 其他输入保证合法，无需其他校验。All other inputs are valid; no other validation required.
# 返回 / Return: CPU tensor [1,P+H1*W1,D], same dtype; prefix bits unchanged.
# 空间 token 按行主序还原/展平，每个 D 通道独立；不得修改输入。
# Reshape/flatten spatial tokens in row-major order, independently per D channel.
# Do not modify the input. 输出 dtype 与输入一致，prefix 必须逐位保持。
# old_hw==new_hw 时直接 clone（不得别名返回）。Equal sizes: return a clone, not an alias.
# 显式实现 half-pixel Keys bicubic：u=(i+0.5)*src/dst-0.5，不 clamp u；
# floor(u)+[-1,0,1,2] 的索引分别 clamp，权重按 clamp 前的距离计算。
# Implement half-pixel Keys bicubic explicitly: u=(i+0.5)*src/dst-0.5, never clamp u;
# clamp each index floor(u)+[-1,0,1,2] separately; use pre-clamp distances for weights.
# a=-0.75，4x4 邻点，无 antialias、无额外 offset；坐标、权重、累加采用 float64 后转回。
# Use a=-0.75 and 4x4 taps, no AA/offset; compute coordinates, weights, sums in float64, then cast back.
# 禁止调用 interpolate/grid_sample/upsample（含别名/封装）；只准测试 oracle 使用。
# Do not call interpolate/grid_sample/upsample, including aliases/wrappers; oracle only.
# 空间输出按 atol=rtol=1e-5(float32)/1e-10(float64) 比较；prefix 必须逐位不变。
# Spatial tolerance: atol=rtol=1e-5 (float32), 1e-10 (float64); prefixes must be bit-exact.
# 无需反向传播。No backward pass required.
def solve(pos_embed, old_hw, new_hw, num_prefix_tokens=1):
    """隔离 prefix 后重采样空间位置 / Resample spatial positions, preserving prefixes."""
    raise NotImplementedError
