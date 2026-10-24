import torch


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_example(solve):
    shards = [torch.tensor([10., 11.]), torch.tensor([20., 21.]), torch.tensor([30., 31.])]
    out = solve(shards)
    full = torch.tensor([10., 11., 20., 21., 30., 31.])
    assert isinstance(out, list) and len(out) == 3, "应返回每 rank 一份结果的列表"
    for r in out:
        assert torch.equal(r, full), r.tolist()


def check_rank_order(solve):
    torch.manual_seed(42)
    shards = [torch.randn(4) for _ in range(5)]
    out = solve(shards)
    full = torch.cat(shards, dim=0)
    assert len(out) == 5
    for r in out:
        assert torch.equal(r, full), "拼接须严格按 rank 顺序 0..W-1"


def check_single_rank(solve):
    s = [torch.tensor([1., 2., 3.])]
    out = solve(s)
    assert len(out) == 1 and torch.equal(out[0], torch.tensor([1., 2., 3.]))


def check_2d_shards_concat_dim0(solve):
    # 分片是二维时沿 dim0 拼接（每 rank 的行块）
    a = torch.arange(6.).reshape(3, 2)
    b = torch.arange(6., 12.).reshape(3, 2)
    out = solve([a, b])
    full = torch.cat([a, b], dim=0)
    assert len(out) == 2
    for r in out:
        assert torch.equal(r, full) and r.shape == (6, 2), r.shape


def check_input_not_mutated(solve):
    shards = [torch.randn(3) for _ in range(4)]
    snapshot = [t.clone() for t in shards]
    solve(shards)
    for a, b in zip(shards, snapshot):
        assert torch.equal(a, b), "不应修改输入分片"


check_example.example = True
