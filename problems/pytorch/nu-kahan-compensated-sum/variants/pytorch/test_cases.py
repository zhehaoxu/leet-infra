import math

import torch


EX = [1e16, 1.0, -1e16, 1.0]


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_example(solve):
    x = torch.tensor(EX)
    assert solve(x, 'naive') == 1.0
    assert solve(x, 'kahan') == 1.0
    assert solve(x, 'kahan_babuska') == 2.0
    assert solve(x, 'pairwise', 8) == 1.0


def check_matches_fsum(solve):
    # fsum 是精确（exhaustively carried）求和，作为与参考解无关的 oracle
    torch.manual_seed(0)
    for n in (7, 64, 997):
        x = torch.randn(n)
        want = math.fsum(x.tolist())
        for mode in ('naive', 'kahan', 'kahan_babuska'):
            got = solve(x, mode)
            assert abs(got - want) <= 1e-12 * max(1.0, abs(want)), (mode, n, got, want)


def check_beats_naive(solve):
    # 1.0 加上上万次 1e-16：double 的 ulp(1.0)≈2.2e-16，朴素累加会整体吞掉
    xs = [1.0] + [1e-16] * 10000
    x = torch.tensor(xs)
    want = math.fsum(xs)
    naive = solve(x, 'naive')
    kb = solve(x, 'kahan_babuska')
    assert naive == 1.0
    assert abs(kb - want) <= 1e-15
    assert abs(kb - want) < abs(naive - want)


def check_pairwise_degenerates(solve):
    torch.manual_seed(3)
    x = torch.randn(5, 3)
    n = x.numel()
    assert solve(x, 'pairwise', n) == solve(x, 'naive')
    assert solve(x, 'pairwise', n + 10) == solve(x, 'naive')
    # block=2 时 [1e16+1, -1e16+1] 两个块和相加正好抵消，比 naive 更差
    assert solve(torch.tensor(EX), 'pairwise', 2) == 0.0
    # 良态数据上各块长结果一致且等于精确和
    y = torch.arange(1.0, 21.0)
    for b in (1, 3, 7, 20):
        assert solve(y, 'pairwise', b) == math.fsum(y.tolist())


def check_special_values(solve):
    modes = ('naive', 'kahan', 'kahan_babuska', 'pairwise')
    for mode in modes:
        assert solve(torch.zeros(0), mode) == 0.0
        assert solve(torch.tensor([float('inf'), 1.0]), mode) == float('inf')
        assert math.isnan(solve(torch.tensor([float('inf'), 1.0, float('-inf')]), mode))
        assert math.isnan(solve(torch.tensor([1.0, float('nan'), 2.0]), mode))


def check_invalid_input(solve):
    for bad in (torch.tensor([1, 2, 3]), torch.tensor([True, False])):
        try:
            solve(bad, 'kahan_babuska')
            raise AssertionError('non-float tensor must raise')
        except ValueError:
            pass
    for mode in ('bogus', '', 'Kahan', None, 0):
        try:
            solve(torch.tensor([1.0]), mode)
            raise AssertionError('bad mode must raise')
        except ValueError:
            pass
    for block in (0, -1):
        try:
            solve(torch.tensor([1.0]), 'pairwise', block)
            raise AssertionError('bad block must raise')
        except ValueError:
            pass


check_example.example = True
