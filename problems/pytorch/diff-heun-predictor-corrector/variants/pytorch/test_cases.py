import math

import numpy as np
import torch


def _oracle_half(mo, sample, sigmas, i, state=None, prediction_type="epsilon",
                 clip_sample=False, clip_sample_range=1.0):
    """独立 numpy 重写 Heun 半步（公式从 diffusers scheduling_heun_discrete.py
    的 step L558-659 数学定义直接推出）。"""
    mo_ = np.asarray(mo, dtype=np.float64)
    x = np.asarray(sample, dtype=np.float64)
    s = np.asarray(sigmas, dtype=np.float64)

    def x0_fn(sig_in):
        if prediction_type == "epsilon":
            return x - sig_in * mo_
        if prediction_type == "v_prediction":
            return mo_ * (-sig_in / np.sqrt(sig_in**2 + 1.0)) + x / (sig_in**2 + 1.0)
        return mo_

    if state is None:
        sigma = s[i]
        sigma_next = s[i + 1]
        x0 = x0_fn(sigma)
        if clip_sample:
            x0 = np.clip(x0, -clip_sample_range, clip_sample_range)
        deriv = (x - x0) / sigma
        dt = sigma_next - sigma
        prev = x + deriv * dt
        return prev, x0, {"prev_derivative": deriv, "dt": dt, "sample": x}, deriv
    sigma_next = s[i]
    x0 = x0_fn(sigma_next)
    if clip_sample:
        x0 = np.clip(x0, -clip_sample_range, clip_sample_range)
    deriv = (x - x0) / sigma_next
    avg = (state["prev_derivative"] + deriv) / 2.0
    prev = state["sample"] + avg * state["dt"]
    return prev, x0, None, deriv


def check_hand_example(solve):
    """手算硬编码：sigmas=[8,4,4,1,1,0]（N=3 的成对结构）。"""
    sigmas = torch.tensor([8.0, 4.0, 4.0, 1.0, 1.0, 0.0])
    # 预测半步 step 0
    r0 = solve(torch.tensor([1.0, 0.0]), torch.tensor([2.0, -1.0]), sigmas, 0)
    assert torch.allclose(r0["pred_original_sample"], torch.tensor([-6.0, -1.0])), r0
    assert torch.allclose(
        r0["prev_sample"], torch.tensor([-2.0, -1.0])
    ), r0["prev_sample"]
    st = r0["state"]
    assert st is not None
    assert torch.allclose(st["prev_derivative"], torch.tensor([1.0, 0.0], dtype=torch.float64))
    assert abs(st["dt"] - (-4.0)) < 1e-12
    # 校正半步 step 1：新点 [-2,-1]、mo'=[0,1]
    r1 = solve(torch.tensor([0.0, 1.0]), torch.tensor([-2.0, -1.0]), sigmas, 1, st)
    assert torch.allclose(r1["pred_original_sample"], torch.tensor([-2.0, -5.0])), r1
    assert torch.allclose(
        r1["prev_sample"], torch.tensor([0.0, -3.0])
    ), r1["prev_sample"]
    assert r1["state"] is None


def check_full_trajectory(solve):
    """N=3 完整 5 次调用轨迹与 oracle 对拍；末步（step 4）只有预测。"""
    sigmas = torch.tensor([8.0, 4.0, 4.0, 1.0, 1.0, 0.0])
    g = torch.Generator().manual_seed(20261031)

    def rand():
        return torch.randn(4, generator=g)

    # 逻辑步 0
    mo, x = rand(), rand()
    got = solve(mo, x, sigmas, 0)
    op, ox0, ost, od = _oracle_half(mo.numpy(), x.numpy(), sigmas.numpy(), 0)
    assert torch.allclose(got["prev_sample"].to(torch.float64), torch.tensor(op), atol=1e-9)
    assert torch.allclose(got["pred_original_sample"].to(torch.float64), torch.tensor(ox0), atol=1e-9)

    # 逻辑步 0 校正
    mo2 = rand()
    x_star = got["prev_sample"]  # pipeline 传入欧拉半步结果
    got2 = solve(mo2, x_star, sigmas, 1, got["state"])
    op2, ox02, ost2, od2 = _oracle_half(
        mo2.numpy(), x_star.numpy(), sigmas.numpy(), 1, _np_state(got["state"])
    )
    assert torch.allclose(got2["prev_sample"].to(torch.float64), torch.tensor(op2), atol=1e-9)
    assert got2["state"] is None and ost2 is None

    # 逻辑步 1（step 2/3）
    mo, x = rand(), rand()
    got3 = solve(mo, x, sigmas, 2)
    op3, _, ost3, _ = _oracle_half(mo.numpy(), x.numpy(), sigmas.numpy(), 2)
    assert torch.allclose(got3["prev_sample"].to(torch.float64), torch.tensor(op3), atol=1e-9)
    mo2 = rand()
    got4 = solve(mo2, got3["prev_sample"], sigmas, 3, got3["state"])
    op4, _, _, _ = _oracle_half(
        mo2.numpy(), got3["prev_sample"].numpy(), sigmas.numpy(), 3, _np_state(got3["state"])
    )
    assert torch.allclose(got4["prev_sample"].to(torch.float64), torch.tensor(op4), atol=1e-9)

    # 逻辑步 2 = 末步（step 4）：只有预测，σ_next = 0
    mo, x = rand(), rand()
    got5 = solve(mo, x, sigmas, 4)
    op5, ox05, ost5, _ = _oracle_half(mo.numpy(), x.numpy(), sigmas.numpy(), 4)
    assert ost5 is not None and got5["state"] is not None
    assert torch.allclose(got5["prev_sample"].to(torch.float64), torch.tensor(op5), atol=1e-9)
    # 末步预测的 dt = 0 − 1 = −1
    assert abs(got5["state"]["dt"] - (-1.0)) < 1e-12


def _np_state(st):
    return {
        "prev_derivative": st["prev_derivative"].detach().cpu().numpy().astype(np.float64),
        "dt": float(st["dt"]),
        "sample": st["sample"].detach().cpu().numpy().astype(np.float64),
    }


def check_v_prediction_and_clip(solve):
    sigmas = torch.tensor([4.0, 1.0, 1.0, 0.0])  # N=2
    sample = torch.tensor([2.0, -1.0])
    mo = torch.tensor([1.0, 0.5])

    got = solve(mo, sample, sigmas, 0, prediction_type="v_prediction")
    op, ox0, ost, _ = _oracle_half(
        mo.numpy(), sample.numpy(), sigmas.numpy(), 0, prediction_type="v_prediction"
    )
    assert torch.allclose(got["prev_sample"].to(torch.float64), torch.tensor(op), atol=1e-9)
    assert torch.allclose(got["pred_original_sample"].to(torch.float64), torch.tensor(ox0), atol=1e-9)

    # clip_sample 在导数计算之前作用在 x0 上
    big = torch.tensor([10.0, -10.0])
    mo_big = torch.tensor([0.0, 0.0])  # epsilon 下 x0 = sample
    gotc = solve(mo_big, big, sigmas, 0, clip_sample=True, clip_sample_range=0.5)
    # x0 clamp 到 ±0.5 → derivative = (10 − 0.5)/4 = 2.375 与 (−10 + 0.5)/4 = −2.375
    assert torch.allclose(
        gotc["pred_original_sample"], torch.tensor([0.5, -0.5])
    ), gotc["pred_original_sample"]
    assert torch.allclose(
        gotc["prev_sample"].to(torch.float64),
        big.to(torch.float64) + torch.tensor([2.375, -2.375], dtype=torch.float64) * (-3.0),
        atol=1e-9,
    )


def check_validation(solve):
    sigmas = torch.tensor([8.0, 4.0, 4.0, 1.0, 1.0, 0.0])
    st = {
        "prev_derivative": torch.tensor([1.0, 0.0], dtype=torch.float64),
        "dt": -4.0,
        "sample": torch.tensor([2.0, -1.0], dtype=torch.float64),
    }

    bad = [
        (lambda: solve(torch.randn(2), torch.randn(3), sigmas, 0), "形状不一致"),
        (lambda: solve(torch.randn(2), torch.randn(2), torch.tensor([4.0, 0.0]), 0), "sigmas 太短"),
        (lambda: solve(torch.randn(2), torch.randn(2), sigmas, 5), "step_index 越界"),
        (lambda: solve(torch.randn(2), torch.randn(2), sigmas, 1), "预测半步用奇数索引"),
        (lambda: solve(torch.randn(2), torch.randn(2), sigmas, 0, st), "校正半步用偶数索引"),
        (
            lambda: solve(torch.randn(2), torch.randn(2), sigmas, 5, st),
            "校正半步越界（末逻辑步无校正）",
        ),
        (
            lambda: solve(
                torch.randn(2), torch.randn(2),
                torch.tensor([4.0, 1.0, 2.0, 0.0]), 1, st,
            ),
            "成对结构被破坏",
        ),
        (lambda: solve(torch.randn(2), torch.randn(2), sigmas, 0, prediction_type="flow"), "未知类型"),
        (
            lambda: solve(torch.randn(2), torch.randn(2), sigmas, 1, {"dt": -4.0, "sample": st["sample"]}),
            "state 缺键",
        ),
    ]
    for fn, tag in bad:
        try:
            fn()
        except ValueError:
            pass
        else:
            raise AssertionError(f"用例未抛错：{tag}")
