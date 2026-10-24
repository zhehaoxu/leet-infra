import torch


def _ref(grads, loss_scale, growth_interval, consec_ok, backoff=2.0, growth=2.0):
    scaled = (grads * loss_scale).to(torch.float16)
    overflow = not bool(torch.isfinite(scaled).all())
    if overflow:
        return {"overflow": True, "skip_update": True,
                "new_loss_scale": float(loss_scale / backoff),
                "new_consec_ok": 0, "unscaled_grads": grads.clone()}
    will_grow = (consec_ok + 1) >= growth_interval
    new_scale = float(loss_scale * growth if will_grow else loss_scale)
    new_consec = 0 if will_grow else consec_ok + 1
    unscaled = (scaled.float() / loss_scale).to(grads.dtype)
    return {"overflow": False, "skip_update": False, "new_loss_scale": new_scale,
            "new_consec_ok": new_consec, "unscaled_grads": unscaled}


def _check(solve, grads, loss_scale, gi, c, **kw):
    out = solve(grads, loss_scale, gi, c, **kw)
    exp = _ref(grads, loss_scale, gi, c, **kw)
    assert out["overflow"] == exp["overflow"], out
    assert out["skip_update"] == exp["skip_update"], out
    assert abs(out["new_loss_scale"] - exp["new_loss_scale"]) < 1e-6, out
    assert out["new_consec_ok"] == exp["new_consec_ok"], out
    assert torch.allclose(out["unscaled_grads"], exp["unscaled_grads"], atol=1e-9), out


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_example_no_overflow(solve):
    out = solve(torch.tensor([0.001, 0.002]), 65536.0, 2000, 0)
    assert out["overflow"] is False and out["skip_update"] is False
    assert out["new_loss_scale"] == 65536.0 and out["new_consec_ok"] == 1
    assert out["unscaled_grads"].tolist() == [0.0010004043579101562, 0.0020008087158203125], out["unscaled_grads"]


def check_example_overflow(solve):
    out = solve(torch.tensor([1.0, 2.0, 100.0]), 65536.0, 2000, 0)
    assert out["overflow"] is True and out["skip_update"] is True
    assert out["new_loss_scale"] == 32768.0 and out["new_consec_ok"] == 0
    assert out["unscaled_grads"].tolist() == [1.0, 2.0, 100.0]  # 原样返回


def check_growth_trigger(solve):
    out = solve(torch.tensor([0.001, 0.002]), 65536.0, 2, 1)
    assert out["new_loss_scale"] == 131072.0 and out["new_consec_ok"] == 0
    assert out["skip_update"] is False


def check_fp16_max_boundary(solve):
    # 65519 → 舍入到 65504（有限、不上溢）；65520 → inf（上溢）
    assert solve(torch.tensor([1.0]), 65519.0, 2000, 0)["overflow"] is False
    assert solve(torch.tensor([1.0]), 65520.0, 2000, 0)["overflow"] is True


def check_no_growth_when_short(solve):
    out = solve(torch.tensor([0.5]), 64.0, 5, 2)  # c=3 < 5
    assert out["new_loss_scale"] == 64.0 and out["new_consec_ok"] == 3


def check_random_agrees(solve):
    torch.manual_seed(7)
    for _ in range(30):
        g = torch.randn(8) * (10 ** torch.randint(-6, 4, ()).item())
        ls = float(2 ** torch.randint(1, 20, ()).item())
        gi = int(torch.randint(1, 6, ()).item())
        c = int(torch.randint(0, 6, ()).item())
        _check(solve, g, ls, gi, c)


def check_does_not_mutate_input(solve):
    g = torch.tensor([1.0, 2.0, 100.0])
    snap = g.clone()
    solve(g, 65536.0, 2000, 0)
    assert torch.equal(g, snap)


check_example_no_overflow.example = True
check_example_overflow.example = True
