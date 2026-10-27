import math

import torch

# 示例期望值（reduction="mean"，两组手写框的均值；用参考实现实测回填）
EXAMPLE_GIOU = 1.0264471769
EXAMPLE_DIOU = 0.9477209449
EXAMPLE_CIOU = 0.9518885612

# 示例逐对期望值（reduction="none"，pair0 / pair1）
EXAMPLE_NONE_GIOU = [1.0793650150, 0.9735293984]
EXAMPLE_NONE_DIOU = [0.9682539701, 0.9271879196]
EXAMPLE_NONE_CIOU = [0.9682539701, 0.9355232120]

EX_PRED = [[0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 1.0, 2.0]]
EX_TARGET = [[1.0, 1.0, 3.0, 3.0], [0.5, 0.5, 2.5, 2.0]]


def _oracle_pair(p, t):
    """单对框的独立标量重算：python 循环 + math.atan，与向量化解法走完全不同的路径。"""
    px1, py1, px2, py2 = (float(v) for v in p)
    tx1, ty1, tx2, ty2 = (float(v) for v in t)
    iw = max(0.0, min(px2, tx2) - max(px1, tx1))
    ih = max(0.0, min(py2, ty2) - max(py1, ty1))
    inter = iw * ih
    area_p = (px2 - px1) * (py2 - py1)
    area_t = (tx2 - tx1) * (ty2 - ty1)
    union = area_p + area_t - inter
    iou = inter / union if union > 0.0 else 0.0
    ew = max(0.0, max(px2, tx2) - min(px1, tx1))
    eh = max(0.0, max(py2, ty2) - min(py1, ty1))
    c_area = ew * eh
    c2 = ew * ew + eh * eh
    giou = iou - ((c_area - union) / c_area if c_area > 0.0 else 0.0)
    rho2 = ((px1 + px2) / 2.0 - (tx1 + tx2) / 2.0) ** 2 \
        + ((py1 + py2) / 2.0 - (ty1 + ty2) / 2.0) ** 2
    diou = iou - (rho2 / c2 if c2 > 0.0 else 0.0)
    pw, ph = max(px2 - px1, 1e-7), max(py2 - py1, 1e-7)
    tw, th = max(tx2 - tx1, 1e-7), max(ty2 - ty1, 1e-7)
    v = (4.0 / math.pi ** 2) * (math.atan(tw / th) - math.atan(pw / ph)) ** 2
    den = (1.0 - iou) + v
    alpha = v / den if den > 0.0 else 0.0
    ciou = diou - alpha * v
    return 1.0 - giou, 1.0 - diou, 1.0 - ciou


def _oracle(pred, target, reduction):
    n = pred.shape[0]
    vals = [_oracle_pair(pred[i], target[i]) for i in range(n)]
    if reduction == "none":
        return tuple(torch.tensor([v[k] for v in vals], dtype=torch.float32)
                     for k in range(3))
    if reduction == "mean":
        if n == 0:
            return tuple(torch.zeros((), dtype=torch.float32) for _ in range(3))
        return tuple(torch.tensor(sum(v[k] for v in vals) / n, dtype=torch.float32)
                     for k in range(3))
    raise ValueError(reduction)


def _cmp(solve, pred, target, reduction):
    got = solve(pred, target, reduction)
    exp = _oracle(pred, target, reduction)
    assert isinstance(got, tuple) and len(got) == 3, f"应返回 3 元 tuple，got {got!r}"
    for k, name in enumerate(("giou", "diou", "ciou")):
        assert got[k].shape == exp[k].shape, (name, got[k].shape, exp[k].shape)
        assert torch.allclose(got[k], exp[k], atol=1e-5, rtol=1e-5), \
            f"{name}: got {got[k].tolist()} exp {exp[k].tolist()}"
    return got


def _plain_iou_loss(p, t):
    """独立重算 1 - IoU（与 _oracle_pair 互不共享代码），供不等式探针使用。"""
    px1, py1, px2, py2 = (float(v) for v in p)
    tx1, ty1, tx2, ty2 = (float(v) for v in t)
    iw = max(0.0, min(px2, tx2) - max(px1, tx1))
    ih = max(0.0, min(py2, ty2) - max(py1, ty1))
    inter = iw * ih
    union = (px2 - px1) * (py2 - py1) + (tx2 - tx1) * (ty2 - ty1) - inter
    iou = inter / union if union > 0.0 else 0.0
    return 1.0 - iou


def _rand_boxes(seed, n):
    """固定种子的合法随机框：x2>x1、y2>y1，重叠与不相交混合出现。"""
    torch.manual_seed(seed)
    x1 = torch.rand(n) * 4.0
    y1 = torch.rand(n) * 4.0
    w = torch.rand(n) * 3.0 + 0.2
    h = torch.rand(n) * 3.0 + 0.2
    return torch.stack([x1, y1, x1 + w, y1 + h], dim=1)


def check_example(solve):
    pred = torch.tensor(EX_PRED)
    target = torch.tensor(EX_TARGET)
    g, d, c = solve(pred, target)  # 默认 reduction="mean"
    assert abs(g.item() - EXAMPLE_GIOU) < 1e-6, f"giou {g.item()} != {EXAMPLE_GIOU}"
    assert abs(d.item() - EXAMPLE_DIOU) < 1e-6, f"diou {d.item()} != {EXAMPLE_DIOU}"
    assert abs(c.item() - EXAMPLE_CIOU) < 1e-6, f"ciou {c.item()} != {EXAMPLE_CIOU}"
    gn, dn, cn = solve(pred, target, "none")
    assert torch.allclose(gn, torch.tensor(EXAMPLE_NONE_GIOU), atol=1e-6), gn.tolist()
    assert torch.allclose(dn, torch.tensor(EXAMPLE_NONE_DIOU), atol=1e-6), dn.tolist()
    assert torch.allclose(cn, torch.tensor(EXAMPLE_NONE_CIOU), atol=1e-6), cn.tolist()


check_example.example = True


def check_matches_oracle(solve):
    for seed in range(40):
        n = 1 + seed % 7
        pred = _rand_boxes(100 + seed, n)
        target = _rand_boxes(500 + seed, n)
        _cmp(solve, pred, target, "none")
        _cmp(solve, pred, target, "mean")


def check_identical_boxes_all_zero(solve):
    box = torch.tensor([[1.0, 2.0, 5.0, 9.0], [0.0, 0.0, 1.0, 1.0]])
    for red in ("none", "mean"):
        g, d, c = solve(box, box, red)
        for t in (g, d, c):
            assert torch.all(t == 0), f"pred==target 时三损失应全 0，got {t.tolist()}"


def check_disjoint_giou_beats_plain_iou_loss(solve):
    pred = torch.tensor([[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 2.0, 2.0]])
    target = torch.tensor([[5.0, 5.0, 6.0, 6.0], [-4.0, -4.0, -2.0, -2.0]])
    g, d, c = solve(pred, target)
    plain = torch.tensor([_plain_iou_loss(pred[i], target[i]) for i in range(2)])
    assert torch.all(g >= plain), (g.tolist(), plain.tolist())
    # 不相交时 IoU=0、plain loss=1，GIoU 惩罚项 (C面积-union)/C面积 严格为正 -> loss_giou > 1
    assert torch.all(g > 1.0), g.tolist()
    assert torch.allclose(c, d, atol=1e-6)  # 两组框都是正方形，v=0 -> ciou 退化成 diou


def check_variant_ordering_random(solve):
    for seed in range(30):
        n = 4 + seed % 5
        pred = _rand_boxes(2000 + seed, n)
        target = _rand_boxes(3000 + seed, n)
        g, d, c = solve(pred, target)
        plain = torch.tensor([_plain_iou_loss(pred[i], target[i]) for i in range(n)])
        assert torch.all(c >= d - 1e-6), f"ciou>=diou 违例: {(c - d).tolist()}"
        assert torch.all(d >= plain - 1e-6), f"diou>=1-IoU 违例: {(d - plain).tolist()}"
        assert torch.all(g >= plain - 1e-6), f"giou>=1-IoU 违例: {(g - plain).tolist()}"
        _cmp(solve, pred, target, "none")


def check_reduction_modes(solve):
    pred = _rand_boxes(77, 6)
    target = _rand_boxes(78, 6)
    g, d, c = solve(pred, target, "mean")
    assert g.shape == () and d.shape == () and c.shape == (), "mean 应返回三个标量张量"
    gn, dn, cn = solve(pred, target, "none")
    assert gn.shape == (6,) and dn.shape == (6,) and cn.shape == (6,), "none 应返回三个 (N,) 张量"
    for a, b in ((g, gn), (d, dn), (c, cn)):
        assert torch.allclose(a, b.mean(), atol=1e-6)
    try:
        solve(pred, target, "sum")
    except ValueError:
        return
    raise AssertionError("非法 reduction 应抛 ValueError")


def check_empty_input(solve):
    pred = torch.zeros(0, 4)
    target = torch.zeros(0, 4)
    g, d, c = solve(pred, target, "none")
    assert g.shape == (0,) and d.shape == (0,) and c.shape == (0,)
    g2, d2, c2 = solve(pred, target, "mean")
    for t in (g2, d2, c2):
        assert t.shape == () and t.item() == 0.0, "N=0 时 mean 约定返回 0"
