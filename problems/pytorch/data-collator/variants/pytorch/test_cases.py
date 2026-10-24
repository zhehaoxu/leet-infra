import torch


def _ref(seqs, pad_id, ignore):
    """独立构造期望（与 solution 的填充写法不同的推导路径）。"""
    B = len(seqs)
    L = max(len(s) for s in seqs)
    ii = [[pad_id] * L for _ in range(B)]
    am = [[0] * L for _ in range(B)]
    lb = [[ignore] * L for _ in range(B)]
    for i, s in enumerate(seqs):
        for j, tok in enumerate(s):
            ii[i][j] = tok
            am[i][j] = 1
            lb[i][j] = tok
    return (
        torch.tensor(ii, dtype=torch.long),
        torch.tensor(am, dtype=torch.long),
        torch.tensor(lb, dtype=torch.long),
    )


def _assert(out, seqs, pad_id=-1, ignore=-100):
    eii, eam, elb = _ref(seqs, pad_id, ignore)
    for key, exp in (("input_ids", eii), ("attention_mask", eam), ("labels", elb)):
        assert key in out, f"缺少 {key}"
        got = out[key]
        assert isinstance(got, torch.Tensor), f"{key} 不是张量"
        assert got.dtype == torch.long, f"{key} dtype={got.dtype}"
        assert got.shape == exp.shape, f"{key} shape {got.shape} != {exp.shape}"
        assert torch.equal(got, exp), f"{key} 值不符:\n{got}\n期望:\n{exp}"


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_example(solve):
    out = solve([[1, 2, 3], [4], [5, 6]], 0)
    assert out["input_ids"].tolist() == [[1, 2, 3], [4, 0, 0], [5, 6, 0]], out["input_ids"]
    assert out["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0], [1, 1, 0]]
    assert out["labels"].tolist() == [[1, 2, 3], [4, -100, -100], [5, 6, -100]]


def check_mixed_lengths(solve):
    seqs = [[7, 8, 9, 10], [1], [2, 3], [4, 5, 6]]
    out = solve(seqs, 0)
    _assert(out, seqs, 0, -100)


def check_all_equal_length_no_pad(solve):
    seqs = [[1, 2], [3, 4], [5, 6]]
    out = solve(seqs, 0)
    _assert(out, seqs, 0, -100)
    assert out["attention_mask"].sum().item() == 6


def check_single_seq(solve):
    seqs = [[42, 43]]
    out = solve(seqs, 0)
    _assert(out, seqs, 0, -100)


def check_custom_pad_and_ignore(solve):
    seqs = [[5, 6, 7], [8]]
    out = solve(seqs, -1, ignore_index=0)
    _assert(out, seqs, -1, 0)


def check_labels_pad_is_ignore(solve):
    # 专门盯 labels 的 pad 段：真实段=input_ids 真实段，pad 段=ignore
    seqs = [[10, 11, 12], [13]]
    out = solve(seqs, 0)
    labels = out["labels"]
    assert labels[0].tolist() == [10, 11, 12]
    assert labels[1].tolist() == [13, -100, -100], labels[1].tolist()


def check_does_not_mutate_input(solve):
    seqs = [[1, 2, 3], [4]]
    snapshot = [list(s) for s in seqs]
    solve(seqs, 0)
    assert seqs == snapshot


check_example.example = True
