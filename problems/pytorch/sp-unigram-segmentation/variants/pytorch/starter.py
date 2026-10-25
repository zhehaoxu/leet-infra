import torch


def solve(text, vocab, log_probs):
    """Unigram 最优切分（Viterbi DP）：在所有完整覆盖 text 的子词切分中取 Σ log_prob 最大者，
    返回 int64 一维张量（选中子词在 vocab 中的下标）。无法完整覆盖时抛 ValueError。"""
    # TODO
    raise NotImplementedError
