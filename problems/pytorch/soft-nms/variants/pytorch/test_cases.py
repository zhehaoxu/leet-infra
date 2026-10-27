import math

import torch

# 示例期望值：手写 3 框 + 分数，method="linear" 全默认参数（实测回填）
EX_BOXES = [[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0], [50.0, 50.0, 60.0, 60.0]]
EX_SCORES = [0.9, 0.8, 0.7]
EXAMPLE_INDICES = [0, 2, 1]                 # 实测回填
EXAMPLE_SCORES = [0.8999999762, 0.6999999881, 0.2554621696]  # 实测回填


def _py_iou(a, b):
    """独立 python 标量 IoU：与解法的向量化实现走完全不同的路径。"""
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0.0 else 0.0


def _oracle(boxes, scores, method, sigma, iou_thr, score_thr):
    """独立 python 列表循环 oracle：逐步选择 + 逐步衰减，不与解法共享任何代码。"""
    n = len(scores)
    cur = [float(s) for s in scores]
    alive = [True] * n
    keep_i, keep_s = [], []
    while True:
        best = -1
        for i in range(n):  # 从小到大扫描，并列时自然取最小下标
            if alive[i] and (best == -1 or cur[i] > cur[best]):
                best = i
        if best == -1:
            break
        alive[best] = False
        keep_i.append(best)
        keep_s.append(cur[best])
        for j in range(n):
            if not alive[j]:
                continue
            iou = _py_iou(boxes[best], boxes[j])
            if method == "linear":
                if iou > iou_thr:
                    cur[j] *= 1.0 - iou
            else:
                cur[j] *= math.exp(-(iou * iou) / sigma)
        for j in range(n):
            if alive[j] and cur[j] < score_thr:
                alive[j] = False
    return keep_i, keep_s


def _cmp(solve, boxes, scores, method="linear", sigma=0.5, iou_thr=0.3, score_thr=0.001):
    got_idx, got_sc = solve(torch.tensor(boxes), torch.tensor(scores),
                            method=method, sigma=sigma, iou_thr=iou_thr,
                            score_thr=score_thr)
    exp_i, exp_s = _oracle(boxes, scores, method, sigma, iou_thr, score_thr)
    assert got_idx.dtype == torch.long and got_idx.shape == (len(exp_i),), \
        (got_idx.dtype, got_idx.shape)
    assert got_idx.tolist() == exp_i, (got_idx.tolist(), exp_i)
    assert torch.allclose(got_sc, torch.tensor(exp_s), atol=1e-5, rtol=1e-5), \
        (got_sc.tolist(), exp_s)
    return got_idx, got_sc


def _rand_boxes(seed, n):
    """固定种子的合法随机框：x2>x1、y2>y1，重叠与不相交混合出现。"""
    torch.manual_seed(seed)
    x1 = torch.rand(n) * 4.0
    y1 = torch.rand(n) * 4.0
    w = torch.rand(n) * 3.0 + 0.2
    h = torch.rand(n) * 3.0 + 0.2
    return torch.stack([x1, y1, x1 + w, y1 + h], dim=1)


def check_example(solve):
    got_idx, got_sc = solve(torch.tensor(EX_BOXES), torch.tensor(EX_SCORES))
    assert got_idx.tolist() == EXAMPLE_INDICES, got_idx.tolist()
    assert torch.allclose(got_sc, torch.tensor(EXAMPLE_SCORES), atol=1e-6), got_sc.tolist()


check_example.example = True


def check_matches_oracle_random(solve):
    for seed in range(12):
        n = 2 + seed % 7
        boxes = _rand_boxes(100 + seed, n).tolist()
        scores = (0.2 + 0.8 * torch.rand(n, generator=torch.Generator().manual_seed(400 + seed))).tolist()
        _cmp(solve, boxes, scores, method="linear")
        _cmp(solve, boxes, scores, method="gaussian")
        _cmp(solve, boxes, scores, method="linear", sigma=0.3, iou_thr=0.55, score_thr=0.05)
        _cmp(solve, boxes, scores, method="gaussian", sigma=0.25, score_thr=0.05)


def check_disjoint_identity(solve):
    # 两两互不相交：分数绝不衰减，按原分数降序全保留（并列取下标序）
    boxes = [[0.0, 0.0, 1.0, 1.0], [10.0, 10.0, 12.0, 12.0],
             [5.0, 0.0, 6.0, 1.0], [20.0, 0.0, 21.0, 1.0]]
    scores = [0.5, 0.9, 0.5, 0.7]
    idx, sc = solve(torch.tensor(boxes), torch.tensor(scores))
    assert idx.tolist() == [1, 3, 0, 2], idx.tolist()  # 0.9, 0.7, 0.5(下标0), 0.5(下标2)
    assert torch.equal(sc, torch.tensor([0.9, 0.7, 0.5, 0.5])), sc.tolist()
    # 与 oracle 一致性兜底
    _cmp(solve, boxes, scores)


def check_linear_threshold_boundary(solve):
    # A=[0,0,2,2]、B=[0,0,2,1]：IoU 恰为 0.5（inter=2, union=4）
    boxes = [[0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 2.0, 1.0]]
    scores = [1.0, 0.8]
    # iou_thr=0.5：IoU 不严格大于阈值 -> 不衰减
    idx, sc = solve(torch.tensor(boxes), torch.tensor(scores), iou_thr=0.5)
    assert idx.tolist() == [0, 1] and torch.allclose(sc, torch.tensor([1.0, 0.8]), atol=1e-6)
    # iou_thr=0.3：IoU=0.5 > 0.3 -> s ← s·(1−0.5)=0.4
    idx, sc = solve(torch.tensor(boxes), torch.tensor(scores), iou_thr=0.3)
    assert idx.tolist() == [0, 1] and torch.allclose(sc, torch.tensor([1.0, 0.4]), atol=1e-6)
    # 阈值上沿：iou_thr=0.499999 时衰减发生
    _, sc2 = solve(torch.tensor(boxes), torch.tensor(scores), iou_thr=0.499999)
    assert sc2[1].item() < 0.8, sc2.tolist()


def check_gaussian_no_hard_threshold(solve):
    # 微重叠框：IoU≈0.005 远小于任何线性阈值，gaussian 仍必须衰减
    boxes = [[0.0, 0.0, 10.0, 10.0], [9.9, 0.0, 19.9, 10.0], [30.0, 30.0, 31.0, 31.0]]
    scores = [1.0, 0.8, 0.5]
    idx, sc = solve(torch.tensor(boxes), torch.tensor(scores), method="gaussian", sigma=0.5)
    iou = _py_iou(boxes[0], boxes[1])
    assert idx.tolist() == [0, 1, 2], idx.tolist()
    assert sc[1].item() < 0.8, "只要重叠就必须衰减"
    assert abs(sc[1].item() - 0.8 * math.exp(-(iou * iou) / 0.5)) < 1e-6, sc[1].item()
    assert abs(sc[2].item() - 0.5) < 1e-7, "不相交框 gaussian 衰减因子恰为 1，分数不变"
    # 对照：同一组数据 linear(iou_thr=0.3) 因 IoU<=阈值 而完全不衰减
    _, sc_lin = solve(torch.tensor(boxes), torch.tensor(scores), method="linear", iou_thr=0.3)
    assert torch.allclose(sc_lin, torch.tensor(scores), atol=1e-7), sc_lin.tolist()


def check_all_culled_extreme(solve):
    # 除首轮最高分框外全被 score_thr 剔光
    boxes = [[0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 2.0, 1.0]]
    scores = [0.5, 0.4]
    idx, sc = solve(torch.tensor(boxes), torch.tensor(scores),
                    method="linear", iou_thr=0.3, score_thr=0.35)
    assert idx.tolist() == [0], idx.tolist()
    assert sc.tolist() == [0.5] or torch.allclose(sc, torch.tensor([0.5]), atol=1e-7)
    # 多框版本：链式重叠逐一衰减后全部低于阈值
    boxes3 = [[0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 2.0, 1.0], [0.0, 0.0, 1.0, 2.0]]
    scores3 = [0.5, 0.45, 0.42]
    idx3, sc3 = solve(torch.tensor(boxes3), torch.tensor(scores3),
                      method="linear", iou_thr=0.1, score_thr=0.3)
    assert idx3.tolist() == [0] and torch.allclose(sc3, torch.tensor([0.5]), atol=1e-7)


def check_degenerate_inputs_and_errors(solve):
    # N=0：两个 (0,) 空张量
    idx, sc = solve(torch.zeros(0, 4), torch.zeros(0))
    assert idx.shape == (0,) and idx.dtype == torch.long
    assert sc.shape == (0,) and sc.dtype == torch.float32
    # N=1：原样保留
    idx, sc = solve(torch.tensor([[3.0, 4.0, 7.0, 9.0]]), torch.tensor([0.42]))
    assert idx.tolist() == [0] and torch.allclose(sc, torch.tensor([0.42]), atol=1e-7)
    # method 非法抛 ValueError
    try:
        solve(torch.zeros(1, 4), torch.zeros(1), method="bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("非法 method 应抛 ValueError")
    # gaussian 且 sigma<=0 抛 ValueError
    for bad_sigma in (0.0, -1.0):
        try:
            solve(torch.zeros(1, 4), torch.zeros(1), method="gaussian", sigma=bad_sigma)
        except ValueError:
            pass
        else:
            raise AssertionError(f"sigma={bad_sigma} 且 method=gaussian 应抛 ValueError")
    # linear 时 sigma 再离谱也不报错（被完全忽略）
    solve(torch.zeros(1, 4), torch.zeros(1), method="linear", sigma=-5.0)
