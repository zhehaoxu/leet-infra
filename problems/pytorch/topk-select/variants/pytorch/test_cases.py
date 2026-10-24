import torch


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_basic(solve):
    x = torch.tensor([[3.0, 1, 7, 2, 9], [5.0, 8, 0, 6, 4]])
    v, i = solve(x, 3)
    assert v.tolist() == [[9.0, 7.0, 3.0], [8.0, 6.0, 5.0]], v.tolist()
    assert i.tolist() == [[4, 2, 0], [1, 3, 0]], i.tolist()


def check_matches_topk(solve):
    torch.manual_seed(0)
    x = torch.randn(4, 10)
    v, i = solve(x, 5)
    rv, ri = torch.topk(x, 5, dim=-1, sorted=True)
    assert torch.equal(v, rv) and torch.equal(i, ri)


def check_k_equals_full(solve):
    # k = 最后一维长度时，退化为整行降序排序
    torch.manual_seed(1)
    x = torch.randn(3, 6)
    v, _ = solve(x, 6)
    assert torch.equal(v, x.sort(-1, descending=True)[0])


def check_gather_consistency(solve):
    # indices 处取出的值应等于 values
    torch.manual_seed(2)
    x = torch.randn(2, 3, 8)
    v, i = solve(x, 4)
    got = torch.gather(x, -1, i)
    assert torch.equal(got, v), "indices 与 values 必须对应同一元素"


check_basic.example = True
