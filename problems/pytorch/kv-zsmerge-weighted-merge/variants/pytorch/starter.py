import torch


def solve(keys, values, scores, counts, dense, new_key, new_value, new_score, metric="l2"):
    """ZeroMerge 零丢弃加权合并（arXiv 2503.10714；上游 mergekv/mergekv.py::cache_merge）。
    keys/values:(T,H,D)、scores/counts:(T,H) 同 dtype（float32/float64）；dense:(T,) bool 掩码。
    i = 非 dense 区分数最低的槽位（同分取小号）；j = dense 区里离 k_i 最近的槽位
    （metric="l2" 取平方欧氏距离，metric="dot_product" 取负点积；同分取小号）。
    把 i / j 合并写进 j：k=(w_i k_i + w_j k_j)/(w_i+w_j)，v 用同一套权重，
    counts[j]=w_i+w_j，scores[j]=s_i+s_j；槽位 i 由新 token 顶替（counts=1, scores=new_score）。
    dense 全 False 时不合并（j=-1），全局最低分槽位直接被顶替；dense 全 True 抛 ValueError。
    返回 dict：keys / values / counts / scores / merged_slots(H,2) int64。内部 float64，输出输入 dtype。"""
    # TODO
    raise NotImplementedError
