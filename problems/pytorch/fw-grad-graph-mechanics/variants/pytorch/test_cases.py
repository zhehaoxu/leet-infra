import random

import torch


def _values(ledger):
    return {
        "w_grad_first": ledger["w_grad_first"].tolist(),
        "w_grad_second": ledger["w_grad_second"].tolist(),
        "d_grad": ledger["d_grad"].tolist(),
    }


def check_anchor(solve):
    x = torch.tensor([1.0, 2.0])
    led = solve(x)
    assert led["w_is_leaf"] is True
    assert led["y_is_leaf"] is False
    assert led["y_grad_fn"] == "AddBackward0"
    assert led["chain"] == ["SumBackward0", "AddBackward0", "MulBackward0", "AccumulateGrad"]
    assert led["leaf_accumulator"] == "AccumulateGrad"
    assert led["d_is_leaf"] is True
    assert led["d_requires_grad"] is False
    assert led["d_shared_storage"] is True
    assert led["n_requires_grad"] is False
    assert led["t_requires_grad"] is False
    assert led["t_is_inference"] is True
    assert led["t_rg_error"] is True
    # dy/dw = 2，backward(ones) 每次注入 [1,1]；两次累加 -> [4,4]
    vals = _values(led)
    assert vals["w_grad_first"] == [2.0, 2.0]
    assert vals["w_grad_second"] == [4.0, 4.0]
    # d*3 -> d.grad = 3*ones
    assert vals["d_grad"] == [3.0, 3.0]


def check_input_untouched(solve):
    rng = random.Random(646)
    for _ in range(20):
        x = torch.tensor([round(rng.uniform(-5, 5), 3) for _ in range(rng.randint(1, 6))])
        before = x.clone()
        led = solve(x)
        assert x.requires_grad is False, "input must stay a non-gradient tensor"
        assert torch.equal(x, before), "input values must not change"
        assert x.is_leaf is True
        # detach 会让 w 与 x 共享存储，等于把入参挂进图——必须 clone
        assert led["w_copies_input"] is True, "w must be a copy, not a detached view of x"


def check_value_independent_grads(solve):
    # dy/dw=2 与 w 的数值无关：任意输入梯度账恒为常数
    rng = random.Random(1646)
    for _ in range(15):
        x = torch.tensor([round(rng.uniform(-20, 20), 4) for _ in range(rng.randint(1, 5))])
        led = solve(x)
        vals = _values(led)
        n = x.numel()
        assert vals["w_grad_first"] == [2.0] * n
        assert vals["w_grad_second"] == [4.0] * n
        assert vals["d_grad"] == [3.0] * n


def check_chain_matches_manual_walk(solve):
    # 链账须与对 (w*2+1).sum() 手动走 next_functions 完全一致
    x = torch.tensor([0.5, -0.5, 3.0])
    led = solve(x)
    w = x.clone().requires_grad_(True)
    loss = (w * 2 + 1).sum()
    manual = []
    node = loss.grad_fn
    while node is not None:
        manual.append(type(node).__name__)
        node = node.next_functions[0][0] if node.next_functions else None
    assert led["chain"] == manual
    assert manual[-1] == "AccumulateGrad"


def check_no_grad_semantics(solve):
    # no_grad 产物不是 inference tensor，且可以之后重新开梯度
    x = torch.tensor([1.0, 2.0])
    led = solve(x)
    assert led["n_requires_grad"] is False
    w = x.clone().requires_grad_(True)
    with torch.no_grad():
        n = w * 3
    assert n.is_inference() is False
    n.requires_grad_(True)
    assert n.is_leaf and n.requires_grad


def check_inference_mode_semantics(solve):
    # inference 产物禁止事后开梯度；no_grad 产物不受此限制
    x = torch.tensor([1.0])
    led = solve(x)
    assert led["t_is_inference"] is True and led["t_rg_error"] is True
    with torch.no_grad():
        m = x * 1
    m.requires_grad_(True)  # 对照：no_grad 产物可以
    assert m.requires_grad


def check_detach_shares_storage(solve):
    x = torch.tensor([1.0, 2.0])
    led = solve(x)
    assert led["d_shared_storage"] is True
    y = x * 2 + 1
    d = y.detach()
    assert d.data_ptr() == y.data_ptr()
    assert d.is_leaf and d.requires_grad is False


def check_noncontiguous_input(solve):
    base = torch.arange(10, dtype=torch.float32)[::2]
    assert not base.is_contiguous()
    led = solve(base)
    vals = _values(led)
    assert vals["w_grad_second"] == [4.0, 4.0, 4.0, 4.0, 4.0]


def check_invalid_inputs(solve):
    for bad in (
        torch.tensor([[1.0, 2.0]]),
        torch.tensor([1, 2, 3]),
    ):
        try:
            solve(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {bad!r}")
    try:
        solve("not a tensor")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for non-tensor")
