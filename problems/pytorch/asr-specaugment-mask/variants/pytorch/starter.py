# SpecAugment mask_along_axis 段生成与合并账
#
# FunASR `funasr/models/specaug/mask_along_axis.py` 的 `mask_along_axis`
# （espnet 同源，Paper: SpecAugment 1904.08779）对 (T, F) 声学谱做 time/freq
# 轴的连续段置零。官方用随机采样生成段，本题把段的 (起点, 宽度) 显式入参，
# 考段生成、布尔并合并与填充值的确定性账。
#
# 实现 solve(spec, masks, axis="time", replace_with_zero=True)：
#
#   输入
#     spec               list[list[float]]  (T, F) 声学谱
#     masks              list[[pos, width]] 段列表（官方每段的随机采样结果）
#     axis               "time" 沿 T 轴（整行置值）| "freq" 沿 F 轴（整列置值）
#     replace_with_zero  True 填 0.0；False 填**全谱所有元素**的算术平均
#                        （官方 spec.mean()，在填充前对原始 spec 计算，
#                        含将要被 mask 的元素）
#
#   输出 dict：
#     masked  list[list[float]]   置值后的谱（输入不改）
#     segments  list[[start, end]] 布尔并后连续被 mask 的区段，左闭右开、
#                                  按 start 升序（官方 any(dim=1) 布尔并的
#                                  确定性呈现）
#
# 规则（照抄官方实现）：
#   1. 轴维度 D = T（time）或 F（freq）；官方注释明说 spec_lengths 未使用。
#   2. 段区间 [pos, pos + width) 左闭右开：mask[d] = (pos <= d) and
#      (d < pos + width)。width == 0 或整段越界（pos >= D）时该段为空。
#   3. 多段通过布尔并合并：官方对每个候选段生成布尔行后 any(dim=1)，
#      **相邻段（[0,2) 与 [2,4)）与重叠段在布尔向量上自然连成同一区段**；
#      越界部分（d >= pos+width > D）自动截断。
#   4. 填充值对所有被 mask 的位置统一替换；replace_with_zero=False 时的
#      mean 是全谱均值（一个标量），不是逐行/逐列均值。
#
# 官方背景（用于理解，不出题）：官方 mask_pos = randint(0, max(1,
# D - mask_length.max()))——起点上限由**全 batch 最大段宽**决定，是本题
# 入参化掉的部分。
#
# 示例：spec=[[1,2,3],[4,5,6],[7,8,9],[10,11,12]]（T=4, F=3）
#   solve(spec, [[1,2]], "time")
#   → 第 1、2 行整行置 0.0，segments=[[1,3]]
#   solve(spec, [[0,2],[2,2]], "time")
#   → 相邻段并成 [[0,4]]，第 0–3 行全零
