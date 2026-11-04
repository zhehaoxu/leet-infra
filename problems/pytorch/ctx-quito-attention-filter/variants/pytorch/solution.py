"""QUITO (arXiv:2408.00274, CCIR 2024) 查询引导注意力压缩 — 过滤规则复现。

官方仓 Wenshansilvia/attention_compressor（quito/utils.py）。注意力打分依赖
HF 模型前向（trigger token + chat template + 最后一层注意力按头平均后取末行
再 softmax），判题环境离线入参化：per-token 注意力分数直接给定，本题钉
打分之后的全部确定性过滤账：

1. reconstruct          子词聚合：Ġ/▁ 前缀开新词，续词 max 聚合、cnt 自增
2. gaussian_smooth      numpy 复现 scipy gaussian_filter1d（mode=reflect、
                        truncate=4.0；sigma=0 恒等）
3. text_filter          词级：按分降序 stable rank，cnt 累积严格 < threshold
                        （sum(cnt)*ratio），溢出词丢弃并立即 break，保原序输出
4. sentence_filter      句级：句分 = 句内词分 max，预算 len(words)*ratio，
                        按分降序贪心、> 预算才断句（等额可过），选完保序
5. dynamic_span         动态句级：breakpoint 句内从最高分词中心，左右交替
                        扩展凑 m 个，还原原序输出

官方缺陷口径（notes 详述）：
- utils.py 未定义 text_filter 引用的 get_sorted_ids（照抄 NameError），
  语义按论文钉为「按注意力降序保留高关注词」的 stable 降序索引；
- 首个无前缀 token 走 else 分支使 cnt 从 1 自增（首词 cnt 偏大 +1）；
- Ċ→换行替换发生在前缀判断之前，替换后的串若不以 Ġ/▁ 开头则整段一词；
- text_filter 严格 <（等额丢弃），sentence_filter 是 >（等额保留），两者相反；
- 全部句子放得下时 breakpoint 未定义（官方进 except 返回 None），钉为 None。
"""
import numpy as np


def reconstruct(tokens, attention):
    """子词序列聚合回词：Ġ/▁ 开头切新词，否则拼入当前词并 max 聚合。"""
    words, attns, cnts = [], [], []
    cur_w, cur_a, cur_c = "", 0.0, 1
    for tok, a in zip(tokens, attention):
        tok = tok.replace("Ċ", "\n")
        if tok.startswith("Ġ") or tok.startswith("▁"):
            if cur_w:
                words.append(cur_w)
                attns.append(cur_a)
                cnts.append(cur_c)
            cur_w, cur_a, cur_c = tok[1:], a, 1
        else:
            cur_w += tok
            cur_a = max(cur_a, a)
            cur_c += 1
    if cur_w:
        words.append(cur_w)
        attns.append(cur_a)
        cnts.append(cur_c)
    return words, attns, cnts


def gaussian_smooth(scores, sigma, truncate=4.0):
    """scipy.ndimage.gaussian_filter1d 的 numpy 复现（mode='reflect'）。"""
    x = np.asarray(scores, dtype=np.float64)
    if sigma == 0:
        return x.copy()
    radius = int(truncate * sigma + 0.5)
    xs = np.arange(-radius, radius + 1, dtype=np.float64)
    phi = np.exp(-0.5 / sigma / sigma * xs * xs)
    phi = phi / phi.sum()
    padded = np.pad(x, radius, mode="symmetric")
    return np.array([np.dot(phi, padded[i:i + len(phi)]) for i in range(len(x))])


def get_sorted_ids(score):
    """按分数降序的 stable 索引（并列按原序）。"""
    return [i for i, _ in sorted(enumerate(score), key=lambda t: -t[1])]


def text_filter(words, score, cnt, ratio):
    """词级过滤：降序贪心，cnt 累积严格小于预算才保留，溢出即断。"""
    rank = get_sorted_ids(score)
    threshold = sum(cnt) * ratio
    keep = [False] * len(words)
    acc = 0
    for i in rank:
        acc += cnt[i]
        if acc < threshold:
            keep[i] = True
        else:
            break
    res_w = [w for w, k in zip(words, keep) if k]
    res_s = [s for s, k in zip(score, keep) if k]
    return res_w, res_s, keep


def sentence_filter(words, attention, sizes, ratio):
    """句级过滤：句分 = max(句内词分)，按分降序贪心，> 预算断句，保序输出。

    sizes 为每句词数（官方经 nltk sent_tokenize 切句，此处离线入参化）。
    全部句子放得下时 breakpoint_seg / budget_left 为 None。
    """
    budget = len(words) * ratio
    segs, scores = [], []
    pos = 0
    for sz in sizes:
        segs.append(words[pos:pos + sz])
        scores.append(max(attention[pos:pos + sz]))
        pos += sz
    assert pos == len(words)
    order = sorted(range(len(segs)), key=lambda i: -scores[i])
    selected, acc = [], 0
    breakpoint_seg, budget_left = None, None
    for i in order:
        if acc + sizes[i] > budget:
            breakpoint_seg = segs[i]
            budget_left = budget - acc
            break
        selected.append(i)
        acc += sizes[i]
    selected.sort()
    out = [" ".join(segs[i]) for i in selected]
    return out, breakpoint_seg, budget_left


def dynamic_span(bp_words, bp_attn, m):
    """breakpoint 句内动态补词：最高分词为中心，左右交替扩展凑 m 个。"""
    n = len(bp_words)
    mid = max(range(n), key=lambda i: bp_attn[i])
    sel = [mid]
    left, right = mid - 1, mid + 1
    while len(sel) < m:
        if left >= 0 and right < n:
            sel.append(left)
            sel.append(right)
            left -= 1
            right += 1
        elif left >= 0:
            sel.append(left)
            left -= 1
        elif right < n:
            sel.append(right)
            right += 1
        else:
            break
    sel.sort()
    return [bp_words[i] for i in sel], [bp_attn[i] for i in sel]


def solve(mode=None, **kw):
    if mode == "reconstruct":
        return reconstruct(kw["tokens"], kw["attention"])
    if mode == "gaussian_smooth":
        return gaussian_smooth(kw["scores"], kw["sigma"])
    if mode == "text_filter":
        return text_filter(kw["words"], kw["score"], kw["cnt"], kw["ratio"])
    if mode == "sentence_filter":
        return sentence_filter(kw["words"], kw["attention"], kw["sizes"], kw["ratio"])
    if mode == "dynamic_span":
        return dynamic_span(kw["bp_words"], kw["bp_attn"], kw["m"])
    raise ValueError(f"unknown mode: {mode}")
