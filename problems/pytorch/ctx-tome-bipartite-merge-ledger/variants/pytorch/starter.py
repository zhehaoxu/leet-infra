CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


# 本题是 ToMe 二分软匹配（bipartite soft matching）合并路径的四段纯数值账：
# 精确余弦比较下的索引计算、gather + scatter_reduce 合并、尺寸加权归一、
# 以及位置回填。不做 ViT 前向、不做 GPU kernel。
#
# mode 列表：
#   match_indices      —— 入参 metric（[B][T][C]，整数）/ r /
#                          class_token / distill_token
#                          返回 r_effective / src_idx / dst_idx / unm_idx /
#                          node_max / num_src / num_dst
#   merge_reduce       —— 入参 x（[B][T][C]）/ src_idx / dst_idx / unm_idx /
#                          reduce（"mean"/"sum"/"amax"）/ distill_token
#                          返回 merged / merged_len / dst_values /
#                          num_src / num_dst
#   size_weighted_merge—— 入参 x（[B][T][C]）/ size（[B][T][1]）/
#                          src_idx / dst_idx / unm_idx / distill_token
#                          返回 merged / merged_size / merged_len
#   unmerge            —— 入参 x（合并后的 [B][T-r][C]）/ src_idx /
#                          dst_idx / unm_idx
#                          返回 out / out_len
#
# 所有 mode 都返回 dict，字段含义见 description.md。
def solve(mode: str, **kw):
    raise NotImplementedError
