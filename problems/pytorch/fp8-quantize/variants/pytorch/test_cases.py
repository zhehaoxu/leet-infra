import torch


def _oracle(x):
    return x.to(torch.float8_e4m3fn).to(x.dtype)


def _assert_exact(solve, x):
    out = solve(x)
    assert out.shape == x.shape and out.dtype == x.dtype
    ref = _oracle(x)
    # 两侧都落到同一 E4M3 格点值（精确相等，转 float64 比对）
    assert torch.equal(out.to(torch.float64), ref.to(torch.float64)), \
        f"\n got {out.tolist()}\n ref {ref.tolist()}"


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_example(solve):
    x = torch.tensor([1.0, 1.06, 1.12, 0.03, 0.015, 2 ** -9, 449.0, 1000.0, -1.06, 448.0])
    out = solve(x)
    expected = torch.tensor([1.0, 1.0, 1.125, 0.029296875, 0.015625,
                             0.001953125, 448.0, 448.0, -1.0, 448.0])
    assert torch.allclose(out, expected, atol=1e-6), out.tolist()


def check_oracle_random_wide(solve):
    torch.manual_seed(0)
    x = (torch.rand(4096) * 2 - 1) * 500   # [-500, 500] 含饱和区
    _assert_exact(solve, x)


def check_oracle_log_grid(solve):
    # 覆盖次正规(2^-11)到大值(2^8)
    x = torch.logspace(-11, 8, 1000, dtype=torch.float64).to(torch.float32)
    _assert_exact(solve, x)


def check_oracle_subnormals(solve):
    x = torch.tensor([2 ** -9, 2 ** -10, 2 ** -11, 1.5 * 2 ** -9, 7.6 * 2 ** -9, 0.001, 0.01])
    _assert_exact(solve, x)


def check_oracle_saturation(solve):
    x = torch.tensor([448.0, 449.0, 463.0, 464.0, 465.0, 480.0, 1e6, -449.0, -1e6])
    out = solve(x)
    _assert_exact(solve, x)
    assert out[6].item() == 448.0 and out[8].item() == -448.0, out.tolist()


def check_oracle_ties_even(solve):
    # 平局点（a/step = k+0.5），验证取偶与 oracle 一致
    # e=0 step=0.125，中点 0.1875(→0.25,偶), 0.3125(→0.25? 2.5→2 →0.25),0.4375(3.5→4→0.5)
    x = torch.tensor([0.0625, 0.1875, 0.3125, 0.4375, 0.5625, 1.0625])
    _assert_exact(solve, x)


def check_zero_and_sign(solve):
    x = torch.tensor([0.0, -0.0, 1e-30, -1e-30])
    out = solve(x)
    assert out[0].item() == 0.0
    assert float(out[2]) >= 0.0 and float(out[3]) <= 0.0


def check_float64(solve):
    torch.manual_seed(1)
    x = (torch.rand(512, dtype=torch.float64) * 2 - 1) * 300
    _assert_exact(solve, x)


def check_does_not_mutate_input(solve):
    x = torch.randn(16) * 10
    snap = x.clone()
    solve(x)
    assert torch.equal(x, snap)


check_example.example = True
