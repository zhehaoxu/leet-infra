import torch


def solve(scores, n_sink, window, k_spec, k_retr, recency_ratio, concentration):
    """FastGen：按注意力画像给每个 head 分类并施加不同保留策略。
    scores:(H,L) float32 非负累计注意力；总分为 0 时两个统计量均记 0。
    统计：R=head 最后 window 列质量占比，C=head 最大列质量占比。
    分类（按序判定）：R>=recency_ratio → 0 streaming；elif C>=concentration → 1 specialized；
    否则 2 retrieval。
    保留：streaming = 前 n_sink ∪ 后 window；specialized = 前 n_sink ∪ 其余 top-k_spec；
    retrieval = 前 n_sink ∪ 其余 top-k_retr（并列取小下标）。
    返回 (head_type:(H,) int64, keep:(H,L) bool, kept:(H,) int64)。
    参数非法抛 ValueError。"""
    # TODO
    raise NotImplementedError
