import torch


def _oracle(weight, tau, grad_out, mode):
    w = weight.flatten().tolist()
    g = grad_out.flatten().tolist()
    mask, grad = [], []
    for wi, gi in zip(w, g):
        a = abs(wi)
        if mode == "hard_ste":
            m = 1.0 if a >= tau else 0.0
            d = gi * m
        else:
            m = min(max(a / tau, 0.0), 1.0)
            d = gi * (m + (a / tau if a < tau else 0.0))
        mask.append(m)
        grad.append(d)
    y = [wi * mi for wi, mi in zip(w, mask)]
    shape = weight.shape
    return (
        torch.tensor(y, dtype=torch.float32).reshape(shape),
        torch.tensor(grad, dtype=torch.float32).reshape(shape),
        torch.tensor(mask, dtype=torch.float32).reshape(shape),
    )


W = torch.tensor([0.5, -1.5, 2.0, -0.25])
G = torch.tensor([1.0, 2.0, 3.0, 4.0])


def check_example(solve):
    y, gw, m = solve(W, 1.0, G, "hard_ste")
    assert m.tolist() == [0.0, 1.0, 1.0, 0.0], m.tolist()
    assert torch.allclose(y, torch.tensor([0.0, -1.5, 2.0, 0.0]), atol=1e-6)
    assert torch.allclose(gw, torch.tensor([0.0, 2.0, 3.0, 0.0]), atol=1e-6)

    y, gw, m = solve(W, 1.0, G, "wsn")
    assert torch.allclose(m, torch.tensor([0.5, 1.0, 1.0, 0.25]), atol=1e-6), m.tolist()
    assert torch.allclose(y, torch.tensor([0.25, -1.5, 2.0, -0.0625]), atol=1e-6), y.tolist()
    assert torch.allclose(gw, torch.tensor([1.0, 2.0, 3.0, 2.0]), atol=1e-6), gw.tolist()


check_example.example = True


def check_matches_oracle(solve):
    torch.manual_seed(31)
    for shape in [(5,), (3, 4), (2, 3, 2)]:
        w = torch.randn(*shape) * 1.5
        g = torch.randn(*shape)
        for tau in (0.5, 1.0, 2.0):
            for mode in ("hard_ste", "wsn"):
                y, gw, m = solve(w, tau, g, mode)
                ey, eg, em = _oracle(w, tau, g, mode)
                assert torch.allclose(m, em, atol=1e-6), f"{mode} {shape} tau={tau} mask"
                assert torch.allclose(y, ey, atol=1e-6), f"{mode} {shape} tau={tau} y"
                assert torch.allclose(gw, eg, atol=1e-5), f"{mode} {shape} tau={tau} grad"


def check_hard_gradient_masks_upstream(solve):
    torch.manual_seed(32)
    w = torch.randn(6, 5)
    g = torch.randn(6, 5)
    _, gw, m = solve(w, 1.0, g, "hard_ste")
    kept = m > 0
    assert torch.all(gw[~kept] == 0), "被剪位置梯度必须为 0"
    assert torch.allclose(gw[kept], g[kept], atol=1e-6), "保留位置梯度应直通"


def check_hard_threshold_inclusive(solve):
    w = torch.tensor([-1.0, 1.0, 0.999, -1.001])
    g = torch.ones(4)
    _, gw, m = solve(w, 1.0, g, "hard_ste")
    assert m.tolist() == [1.0, 1.0, 0.0, 1.0], m.tolist()
    assert gw.tolist() == [1.0, 1.0, 0.0, 1.0], gw.tolist()


def check_wsn_gradient_matches_autograd(solve):
    torch.manual_seed(33)
    w = (torch.randn(7) * 1.3).double().requires_grad_(True)
    g = torch.randn(7).double()
    tau = 0.9
    _, gw, _ = solve(w, tau, g, "wsn")
    y = w * (w.abs() / tau).clamp(0.0, 1.0)
    (auto,) = torch.autograd.grad(y, w, grad_outputs=g)
    assert torch.allclose(gw.double(), auto, atol=1e-10), f"{gw.tolist()} vs {auto.tolist()}"


def check_wsn_mask_saturates(solve):
    w = torch.tensor([0.0, 0.5, 1.0, 10.0])
    g = torch.ones(4)
    _, _, m = solve(w, 1.0, g, "wsn")
    assert torch.allclose(m, torch.tensor([0.0, 0.5, 1.0, 1.0]), atol=1e-6), m.tolist()


def check_wsn_gradient_below_threshold(solve):
    w = torch.tensor([0.2, -0.4, 1.5])
    g = torch.tensor([1.0, 2.0, 3.0])
    _, gw, _ = solve(w, 1.0, g, "wsn")
    # |w|<tau 时 d(w·|w|)/dw = 2|w|/tau
    assert torch.allclose(gw[:2], torch.tensor([0.4, 1.6]), atol=1e-6), gw.tolist()
    assert abs(float(gw[2]) - 3.0) < 1e-6, gw.tolist()


def check_output_is_weight_times_mask(solve):
    torch.manual_seed(34)
    w = torch.randn(4, 4)
    g = torch.randn(4, 4)
    for mode in ("hard_ste", "wsn"):
        y, _, m = solve(w, 0.7, g, mode)
        assert torch.allclose(y, w * m, atol=1e-6), mode
