import torch


def _ref(inputs):
    """独立重算：手写逐元素求和，再按 rank 切片，不复用 solve 的 stack/reshape。"""
    w = len(inputs)
    L = inputs[0].numel()
    total = torch.zeros(L, dtype=torch.float32)
    for t in inputs:
        total = total + t.reshape(L)
    chunk = L // w
    return [total[i * chunk:(i + 1) * chunk].clone() for i in range(w)]


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_example(solve):
    inputs = [torch.tensor([1., 2., 3.]), torch.tensor([4., 5., 6.]), torch.tensor([7., 8., 9.])]
    out = solve(inputs)
    assert len(out) == 3
    assert torch.equal(out[0], torch.tensor([12.])), out[0].tolist()
    assert torch.equal(out[1], torch.tensor([15.])), out[1].tolist()
    assert torch.equal(out[2], torch.tensor([18.])), out[2].tolist()


def check_chunks_rebuild_full_sum(solve):
    torch.manual_seed(42)
    inputs = [torch.randn(6) for _ in range(3)]     # L=6, W=3, chunk=2
    out = solve(inputs)
    full = torch.stack(inputs, 0).sum(0)
    # 各 rank 分段拼回去 == 完整逐元素和
    assert torch.equal(torch.cat(out, dim=0), full), "分片拼接应还原完整归约结果"
    for i, o in enumerate(out):
        assert o.shape == (2,), o.shape


def check_chunk_size_two(solve):
    inputs = [torch.tensor([1., 2., 3., 4.]), torch.tensor([5., 6., 7., 8.])]
    out = solve(inputs)                             # sum=[6,8,10,12], chunk=2
    assert torch.equal(out[0], torch.tensor([6., 8.])), out[0].tolist()
    assert torch.equal(out[1], torch.tensor([10., 12.])), out[1].tolist()


def check_matches_ref(solve):
    for w, L in [(4, 8), (2, 6), (1, 5), (3, 9)]:
        torch.manual_seed(w * 10 + L)
        inputs = [torch.randn(L) for _ in range(w)]
        out = solve(inputs)
        want = _ref(inputs)
        assert len(out) == w
        for o, r in zip(out, want):
            assert torch.equal(o, r), (w, L)


def check_input_not_mutated(solve):
    inputs = [torch.randn(4) for _ in range(2)]
    snapshot = [t.clone() for t in inputs]
    solve(inputs)
    for a, b in zip(inputs, snapshot):
        assert torch.equal(a, b), "不应修改输入张量"


check_example.example = True
