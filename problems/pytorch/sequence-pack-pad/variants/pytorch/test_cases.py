import torch


def _ref(seqs, bs, pad_id):
    """独立重算：先算每条序列长度，逐块填充时按 (doc, offset) 推导，走不同路径。"""
    total = sum(len(s) for s in seqs)
    nb = (total + bs - 1) // bs
    # 展平 (token, pos, seg) 列表，与 solution 的 enumerate 拼接不同：按偏移重建
    flat = []
    for d, s in enumerate(seqs):
        for off in range(len(s)):
            flat.append((s[off], off, d))
    padn = nb * bs - total
    for _ in range(padn):
        flat.append((pad_id, 0, -1))
    ids = [t[0] for t in flat]
    pos = [t[1] for t in flat]
    seg = [t[2] for t in flat]
    am = [0 if seg[k] == -1 else 1 for k in range(len(seg))]
    R = lambda x: torch.tensor(x, dtype=torch.long).view(nb, bs)
    return {"input_ids": R(ids), "position_ids": R(pos),
            "segment_ids": R(seg), "attention_mask": R(am)}


def _assert(out, seqs, bs, pad_id):
    exp = _ref(seqs, bs, pad_id)
    assert set(out.keys()) >= set(exp.keys()), out.keys()
    for k in exp:
        got = out[k]
        assert isinstance(got, torch.Tensor), f"{k} 非张量"
        assert got.dtype == torch.long, f"{k} dtype={got.dtype}"
        assert got.shape == exp[k].shape, f"{k} shape {got.shape}!={exp[k].shape}"
        assert torch.equal(got, exp[k]), f"{k} 不符:\n{got}\n期望:\n{exp[k]}"


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_example(solve):
    out = solve([[10, 11, 12], [20], [30, 31]], 4, 0)
    assert out["input_ids"].tolist() == [[10, 11, 12, 20], [30, 31, 0, 0]], out["input_ids"]
    assert out["position_ids"].tolist() == [[0, 1, 2, 0], [0, 1, 0, 0]]
    assert out["segment_ids"].tolist() == [[0, 0, 0, 1], [2, 2, -1, -1]]
    assert out["attention_mask"].tolist() == [[1, 1, 1, 1], [1, 1, 0, 0]]


def check_no_pad_exact_fit(solve):
    out = solve([[5], [6, 7]], 3, 0)
    _assert(out, [[5], [6, 7]], 3, 0)
    assert out["attention_mask"].sum().item() == 3


def check_position_continues_across_blocks(solve):
    # 单条长序列跨两块：position 越界继续递增，不重置
    out = solve([[1, 2, 3, 4, 5, 6, 7, 8]], 4, 0)
    assert out["position_ids"].tolist() == [[0, 1, 2, 3], [4, 5, 6, 7]], out["position_ids"]
    assert out["segment_ids"].tolist() == [[0, 0, 0, 0], [0, 0, 0, 0]]


def check_random_varied(solve):
    seqs = [[1, 2], [3], [4, 5, 6, 7], [8], [9, 10, 11]]
    out = solve(seqs, 3, -1)
    _assert(out, seqs, 3, -1)


def check_segment_pad_is_minus_one(solve):
    out = solve([[1, 2, 3]], 2, 0)  # total=3 -> nb=2, 1 pad
    assert out["segment_ids"].tolist() == [[0, 0], [0, -1]], out["segment_ids"]
    assert out["attention_mask"].tolist() == [[1, 1], [1, 0]]


def check_does_not_mutate_input(solve):
    seqs = [[1, 2, 3], [4]]
    snap = [list(s) for s in seqs]
    solve(seqs, 4, 0)
    assert seqs == snap


check_example.example = True
