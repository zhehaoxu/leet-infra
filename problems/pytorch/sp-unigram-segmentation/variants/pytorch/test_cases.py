import itertools
import math

import torch


def _oracle(text, vocab, log_probs):
    piece2idx = {p: i for i, p in enumerate(vocab)}
    n = len(text)
    best_sc = float("-inf")
    best_idx = None
    for mask in itertools.product([0, 1], repeat=max(n - 1, 0)):
        parts = []
        start = 0
        for i, cut in enumerate(mask):
            if cut:
                parts.append(text[start:i + 1])
                start = i + 1
        parts.append(text[start:])
        if all(p in piece2idx for p in parts):
            sc = sum(log_probs[piece2idx[p]] for p in parts)
            if sc > best_sc + 1e-12:
                best_sc = sc
                best_idx = [piece2idx[p] for p in parts]
    if best_idx is None:
        raise ValueError("no segmentation")
    return torch.tensor(best_idx, dtype=torch.long)


V1 = ["u", "n", "i", "g", "r", "a", "m", "un", "uni", "gram", "nig", "am"]
P1 = [math.log(p) for p in
      [0.05, 0.05, 0.08, 0.05, 0.05, 0.09, 0.05, 0.1, 0.5, 0.4, 0.01, 0.02]]
V2 = ["a", "b", "c", "ab", "bc", "abc", "ac"]
P2 = [math.log(p) for p in [0.01, 0.01, 0.01, 0.05, 0.02, 0.6, 0.03]]


def check_example(solve):
    out = solve("unigram", V1, P1)
    assert out.tolist() == [8, 9], out.tolist()


check_example.example = True


def check_single_piece(solve):
    out = solve("abc", V2, P2)
    assert out.tolist() == [5], out.tolist()


def check_dp_beats_greedy(solve):
    # 贪心最长会取 "xy"(0.005)，但 DP 选 x*y*z(0.01)——全局最优优于贪婪
    vocab = ["x", "y", "z", "xy"]
    lp = [math.log(p) for p in [0.1, 0.1, 0.1, 0.005]]
    out = solve("xyz", vocab, lp)
    assert out.tolist() == [0, 1, 2], out.tolist()


def check_no_segmentation_raises(solve):
    vocab = ["a", "b", "c"]
    lp = [math.log(0.33)] * 3
    try:
        solve("abz", vocab, lp)  # 'z' 无法覆盖
    except ValueError:
        return
    raise AssertionError("无法覆盖的串应抛 ValueError")


def check_matches_oracle(solve):
    cases = [
        ("unigram", V1, P1),
        ("abc", V2, P2),
        ("gram", V1, P1),
        ("uni", V1, P1),
        ("xyz", ["x", "y", "z", "xy"],
         [math.log(p) for p in [0.1, 0.1, 0.1, 0.005]]),
    ]
    for text, vocab, lp in cases:
        y = solve(text, vocab, lp)
        assert torch.equal(y, _oracle(text, vocab, lp)), text
