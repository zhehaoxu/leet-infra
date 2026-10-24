import torch


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_example(solve):
    out = solve(torch.tensor([1., 2., 3., 4., 5., 6.]), 3)
    assert len(out) == 3
    assert torch.equal(out[0], torch.tensor([1., 2.])), out[0].tolist()
    assert torch.equal(out[1], torch.tensor([3., 4.])), out[1].tolist()
    assert torch.equal(out[2], torch.tensor([5., 6.])), out[2].tolist()


def check_inverse_of_allgather(solve):
    torch.manual_seed(42)
    full = torch.randn(12)
    out = solve(full, 4)                       # chunk=3
    assert len(out) == 4
    assert torch.equal(torch.cat(out, dim=0), full), "各段拼回应还原完整数据"
    for i in range(4):
        assert torch.equal(out[i], full[i * 3:(i + 1) * 3])


def check_chunk_one(solve):
    full = torch.tensor([7., 8., 9.])
    out = solve(full, 3)                       # chunk=1
    for i in range(3):
        assert out[i].shape == (1,) and torch.equal(out[i], full[i:i + 1])


def check_world_one(solve):
    full = torch.tensor([1., 2., 3.])
    out = solve(full, 1)
    assert len(out) == 1 and torch.equal(out[0], full)


def check_matches_ref(solve):
    for L, w in [(8, 2), (9, 3), (16, 4), (6, 6), (5, 1)]:
        torch.manual_seed(L * 7 + w)
        full = torch.randn(L)
        out = solve(full, w)
        chunk = L // w
        assert len(out) == w
        for i in range(w):
            assert torch.equal(out[i], full[i * chunk:(i + 1) * chunk]), (L, w, i)


def check_input_not_mutated(solve):
    full = torch.arange(10.).float()
    snap = full.clone()
    solve(full, 5)
    assert torch.equal(full, snap), "不应修改输入张量"


check_example.example = True
