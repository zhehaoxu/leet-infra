import math

import numpy as np
import torch


KEYS = {"rank_float", "rank_floor", "rank_topup", "rank_aligned", "rank_sum", "stored_ratio"}


def _matrix(groups, head_group_size, lr_group_dims):
    rows = []
    for value in groups:
        rows.extend([[float(value)] * lr_group_dims] * head_group_size)
    return rows


def _oracle(fisher, ratio, head_group_size, lr_group_dims, block_size):
    f = np.asarray(fisher, dtype=np.float64)
    n = f.shape[0] // head_group_size
    group = f.reshape(n, head_group_size, lr_group_dims).mean(axis=(1, 2))
    total_rank = n * lr_group_dims
    target = total_rank * float(ratio)
    rank_float = target * group / group.sum()
    rank_floor = np.minimum(np.floor(rank_float), lr_group_dims)
    order = sorted(range(n), key=lambda g: (rank_float[g] - rank_floor[g], g))
    need = math.floor(target - rank_floor.sum())
    topup = rank_floor.astype(np.int64)
    cursor = 0
    while need > 0:
        g = order[cursor % n]
        cursor += 1
        if topup[g] == lr_group_dims:
            continue
        topup[g] += 1
        need -= 1
    aligned = np.maximum(1, np.round(topup / block_size)).astype(np.int64) * block_size
    return {
        "rank_float": rank_float,
        "rank_floor": rank_floor.astype(np.int64),
        "rank_topup": topup,
        "rank_aligned": aligned,
        "rank_sum": np.int64(aligned.sum()),
        "stored_ratio": aligned.sum() / total_rank,
    }


def _verify(solve, dtype, fisher, ratio, head_group_size, lr_group_dims, block_size):
    tensor = torch.tensor(fisher, dtype=dtype)
    snapshot = tensor.clone()
    got = solve(tensor, ratio, head_group_size, lr_group_dims, block_size)
    want = _oracle(fisher, ratio, head_group_size, lr_group_dims, block_size)
    assert torch.equal(tensor, snapshot), "solve must not write into fisher"
    assert set(got) == KEYS, f"keys {sorted(got)} != {sorted(KEYS)}"
    tol = 1e-5 if dtype == torch.float32 else 2e-9
    assert got["rank_float"].dtype == dtype and got["rank_float"].shape == want["rank_float"].shape
    assert np.allclose(got["rank_float"].to(torch.float64).numpy(), want["rank_float"], atol=tol)
    for key in ("rank_floor", "rank_topup", "rank_aligned"):
        value = got[key]
        assert value.dtype == torch.int64, f"{key} must be int64"
        assert np.array_equal(value.numpy(), want[key]), f"{key}={value.tolist()} != {want[key].tolist()}"
    assert got["rank_sum"].dtype == torch.int64 and got["rank_sum"].shape == ()
    assert int(got["rank_sum"]) == int(want["rank_sum"]), "rank_sum must be the sum of rank_aligned"
    assert got["stored_ratio"].dtype == dtype and got["stored_ratio"].shape == ()
    assert abs(float(got["stored_ratio"]) - float(want["stored_ratio"])) <= tol
    return got


def check_example(solve):
    fisher = _matrix([4.0, 3.0, 2.0, 1.0], 4, 32)
    got = _verify(solve, torch.float64, fisher, 0.5, 4, 32, 8)
    assert np.array_equal(got["rank_float"].numpy(), [25.6, 19.2, 12.8, 6.4])
    assert got["rank_floor"].tolist() == [25, 19, 12, 6]
    assert got["rank_topup"].tolist() == [25, 20, 12, 7]
    assert got["rank_aligned"].tolist() == [24, 16, 16, 8]
    assert int(got["rank_sum"]) == 64 and abs(float(got["stored_ratio"]) - 0.5) < 1e-12
check_example.example = True


def check_smallest_remainder_first(solve):
    fisher = _matrix([7.4, 6.1, 5.9, 4.6], 2, 8)
    got = _verify(solve, torch.float64, fisher, 0.75, 2, 8, 8)
    assert got["rank_floor"].tolist() == [7, 6, 5, 4]
    assert got["rank_topup"].tolist() == [8, 7, 5, 4], "smallest remainders receive the top-up first"


def check_bankers_rounding(solve):
    fisher = _matrix([5.0, 3.0], 1, 32)
    got = _verify(solve, torch.float64, fisher, 0.5, 1, 32, 8)
    assert got["rank_topup"].tolist() == [20, 12]
    assert got["rank_aligned"].tolist() == [16, 16], "20/8=2.5 and 12/8=1.5 both round to the even 2"


def check_zero_fisher_group(solve):
    fisher = _matrix([2.0, 2.0, 0.0], 2, 8)
    got = _verify(solve, torch.float64, fisher, 0.5, 2, 8, 4)
    assert got["rank_float"].tolist() == [6.0, 6.0, 0.0]
    assert got["rank_topup"].tolist() == [6, 6, 0]
    assert got["rank_aligned"].tolist() == [8, 8, 4], "even a zero-Fisher group keeps one block"
    assert int(got["rank_sum"]) == 20, "alignment can push rank_sum above target_rank"


def check_cap_skips_and_cycles(solve):
    fisher = _matrix([100.0, 1.0, 1.0], 1, 8)
    got = _verify(solve, torch.float64, fisher, 1.0, 1, 8, 4)
    assert got["rank_floor"].tolist() == [8, 0, 0], "floors are capped at lr_group_dims"
    assert got["rank_topup"].tolist() == [8, 8, 8], "the capped group is skipped and the rest cycle"


def check_fractional_budget(solve):
    fisher = _matrix([1.0, 1.0, 1.0, 1.0], 1, 32)
    got = _verify(solve, torch.float64, fisher, 0.55, 1, 32, 8)
    assert int(got["rank_topup"].sum()) == 70, "the loop stops once dif <= 0, so it adds floor(dif)"
    assert got["rank_aligned"].tolist() == [16, 16, 16, 16], "alignment can lower rank_sum below target_rank"


def check_dtype_and_random(solve):
    torch.manual_seed(11)
    fisher = torch.rand(12, 24, dtype=torch.float64)
    _verify(solve, torch.float32, fisher.tolist(), 0.4, 3, 24, 8)
    _verify(solve, torch.float64, fisher.tolist(), 0.65, 3, 24, 16)


def check_value_errors(solve):
    base = _matrix([1.0, 2.0], 1, 8)
    cases = [
        (torch.tensor(base, dtype=torch.float32), 0.5, 3, 8, 4),  # rows not divisible
        (torch.tensor(base, dtype=torch.float32), 0.5, 1, 16, 4),  # columns != lr_group_dims
        (torch.tensor(_matrix([0.0, 0.0], 1, 8), dtype=torch.float64), 0.5, 1, 8, 4),
        (torch.tensor(base, dtype=torch.float32), 0.0, 1, 8, 4),  # ratio out of range
        (torch.tensor(base, dtype=torch.float32), 1.5, 1, 8, 4),
        (torch.tensor(base, dtype=torch.float32), 0.5, 1, 8, 0),  # block_size < 1
        (torch.tensor(base, dtype=torch.int64), 0.5, 1, 8, 4),  # not floating point
        (torch.ones(8, dtype=torch.float64), 0.5, 1, 8, 4),  # not 2-D
    ]
    for index, args in enumerate(cases):
        try:
            solve(*args)
        except ValueError:
            continue
        raise AssertionError(f"case {index}: expected ValueError for invalid arguments")
