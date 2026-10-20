"""Test cases for one-hot-encoding."""

import inspect

import torch


def _ref(labels, num_classes):
    """允许在测试参考里使用官方 one_hot。"""
    return torch.nn.functional.one_hot(labels, num_classes).float()


def check_values_match_reference(solve):
    torch.manual_seed(0)
    labels = torch.randint(0, 10, (64,), dtype=torch.int64)
    out = solve(labels, 10)
    assert torch.equal(out, _ref(labels, 10)), "与 F.one_hot(...).float() 不一致"


def check_shape_and_dtype(solve):
    labels = torch.tensor([2, 0, 1, 4])
    out = solve(labels, 5)
    assert out.shape == (4, 5), f"shape {tuple(out.shape)} != (4, 5)"
    assert out.dtype == torch.float32, f"dtype {out.dtype} != float32"


def check_single_hot_per_row(solve):
    torch.manual_seed(1)
    labels = torch.randint(0, 32, (100,), dtype=torch.int64)
    out = solve(labels, 32)
    assert torch.equal(out.sum(dim=1), torch.ones(100)), "存在行不恰好有一个 1"
    assert torch.equal(out.argmax(dim=1), labels), "1 所在列与 label 不一致"


def check_empty_batch(solve):
    labels = torch.empty(0, dtype=torch.int64)
    out = solve(labels, 7)
    assert out.shape == (0, 7), f"空批次 shape {tuple(out.shape)} != (0, 7)"


def check_out_of_range_raises_value_error(solve):
    for bad in (torch.tensor([0, 5]), torch.tensor([-1, 0])):
        try:
            solve(bad, 3)
        except ValueError:
            continue
        raise AssertionError(f"越界标签 {bad.tolist()} 未抛出 ValueError")


def check_boundary_labels(solve):
    labels = torch.tensor([0, 999, 0, 999], dtype=torch.int64)
    out = solve(labels, 1000)
    assert torch.equal(out, _ref(labels, 1000)), "首尾类别边界不一致"
    assert out.data_ptr() != labels.data_ptr(), "输出不得与输入共享存储"


def check_no_builtin_one_hot_in_source(solve):
    try:
        src = inspect.getsource(solve)
    except (OSError, TypeError):
        return  # 拿不到源码（如 REPL），不作强制
    assert "one_hot" not in src, "禁止在你的实现中调用内置 one_hot"


# 样例用例：点「运行」只执行这些；「提交」执行全部用例
check_values_match_reference.example = True
check_shape_and_dtype.example = True
