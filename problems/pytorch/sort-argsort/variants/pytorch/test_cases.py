import torch


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_basic(solve):
    x = torch.tensor([[3.0, 1, 7, 2, 9], [5.0, 8, 0, 6, 4]])
    v, i = solve(x)
    assert v.tolist() == [[1.0, 2, 3, 7, 9], [0.0, 4, 5, 6, 8]], v.tolist()
    assert i.tolist() == [[1, 3, 0, 2, 4], [2, 4, 0, 3, 1]], i.tolist()


def check_matches_sort(solve):
    torch.manual_seed(0)
    x = torch.randn(4, 9)
    v, i = solve(x)
    sv, si = torch.sort(x, dim=-1)
    assert torch.equal(v, sv) and torch.equal(i, si)


def check_argsort_gathers_values(solve):
    # 用返回的 indices 去 gather 原张量，应还原出排序后的 values
    torch.manual_seed(1)
    x = torch.randn(2, 3, 7)
    v, i = solve(x)
    assert torch.equal(torch.gather(x, -1, i), v)


def check_values_ascending(solve):
    torch.manual_seed(2)
    x = torch.randn(5, 6)
    v, _ = solve(x)
    assert bool((v[..., 1:] >= v[..., :-1]).all()), "输出必须升序"


check_basic.example = True
