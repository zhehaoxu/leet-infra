"""CPC (arXiv:2409.01227, AAAI-25) 句级上下文压缩 — 选择规则复现。

官方仓 Workday/cpc（util/preprocessing.py::compress_sample）。编码器打分
（双向注意力编码器 + mean pooling + cos_sim(question 向量)）依赖训练模型，
判题离线入参化：每句的注意力分数、token 长度、boost/fake 标志直接给定，
本题钉打分之后的确定性选择链：

1. boost_flags      re.match（仅句首）决定 boost 优先级（官方 boost_sents_regexp）
2. rank_key         sorted(reverse=True, key=(boost, score))——boost 句最先，
                    其余按分降序；Python stable 排序，并列保原序
3. select_segments  taken 从 -1 起（补偿 +1 空格账），每句 taken += ln + 1，
                    **溢出句先收再 break**（官方 add-then-break 怪癖）、目标恰好
                    等于时继续；fake（question-as-suffix echo）句跳过；
                    输出按原句序过滤（非分数序）
"""
import re


def boost_flags(segments, pattern):
    """官方 re.match 语义：仅句首命中才 boost。pattern None → 全 False。"""
    if pattern is None:
        return [False] * len(segments)
    return [re.match(pattern, s) is not None for s in segments]


def rank_key(boosts, scores):
    """(boost, score) 双键降序 stable 索引：boost 句最先、其余按分降序。"""
    return [i for i, _ in sorted(enumerate(zip(boosts, scores)),
                                 key=lambda t: (t[1][0], t[1][1]), reverse=True)]


def select_segments(segments, scores, boosts, fakes, lengths, target):
    """官方贪心：taken=-1 起、ln+1 计入、溢出句先收再 break、fake 跳过、原序输出。"""
    order = rank_key(boosts, scores)
    indices = set()
    taken = -1
    for i in order:
        if fakes[i]:
            continue
        taken += lengths[i] + 1
        indices.add(i)
        if taken > target:
            break
    out = [seg for i, seg in enumerate(segments) if i in indices]
    return out, indices, taken


def solve(mode=None, **kw):
    if mode == "boost_flags":
        return boost_flags(kw["segments"], kw["pattern"])
    if mode == "rank_key":
        return rank_key(kw["boosts"], kw["scores"])
    if mode == "select_segments":
        out, idxs, taken = select_segments(kw["segments"], kw["scores"], kw["boosts"],
                                           kw["fakes"], kw["lengths"], kw["target"])
        return out, sorted(idxs), taken
    raise ValueError(f"unknown mode: {mode}")
