import math

import torch

# 示例（实测回填）：单层 stride 8，GT=[2,2,22,22]，5 个格点
EXAMPLE_GT = [[2.0, 2.0, 22.0, 22.0]]
EXAMPLE_POINTS = [[4.0, 4.0], [12.0, 4.0], [4.0, 12.0], [12.0, 12.0], [24.0, 12.0]]
EXAMPLE_LEVELS = [0, 0, 0, 0, 0]
EXAMPLE_STRIDES = [8.0]
EXAMPLE_RANGES = [(0.0, 64.0)]
EXAMPLE_LABELS = [0, 0, 0, 0, -1]
EXAMPLE_CENTERNESS = [0.1111111111111111, 0.3333333333333333, 0.3333333333333333, 1.0, 0.0]
EXAMPLE_REG = [[2.0, 2.0, 18.0, 18.0],
               [10.0, 2.0, 10.0, 18.0],
               [2.0, 10.0, 18.0, 10.0],
               [10.0, 10.0, 10.0, 10.0],
               [0.0, 0.0, 0.0, 0.0]]


def check_example(solve):
    gt = torch.tensor(EXAMPLE_GT, dtype=torch.float32)
    pts = torch.tensor(EXAMPLE_POINTS, dtype=torch.float32)
    labels, cen, reg = solve(gt, pts, EXAMPLE_LEVELS, EXAMPLE_STRIDES, EXAMPLE_RANGES)
    assert labels.dtype == torch.int64 and cen.dtype == torch.float32, (labels.dtype, cen.dtype)
    assert labels.tolist() == EXAMPLE_LABELS, labels.tolist()
    assert torch.allclose(cen, torch.tensor(EXAMPLE_CENTERNESS), atol=1e-6), cen.tolist()
    assert torch.allclose(reg, torch.tensor(EXAMPLE_REG), atol=1e-6), reg.tolist()


check_example.example = True


def _oracle(gt, points, levels, strides, ranges, radius=1.5):
    """纯 Python 逐点逐 GT 循环：不建 IoU 矩阵、不做广播。"""
    gl = gt.tolist()
    pl = points.tolist() if hasattr(points, "tolist") else points
    labels, cen, reg = [], [], []
    for (x, y), lv in zip(pl, levels):
        rad = radius * float(strides[int(lv)])
        lo, hi = float(ranges[int(lv)][0]), float(ranges[int(lv)][1])
        best = None
        covered = False
        for gi, (x1, y1, x2, y2) in enumerate(gl):
            l, t, r, b = x - x1, y - y1, x2 - x, y2 - y
            inside = l >= 0 and t >= 0 and r >= 0 and b >= 0
            covered = covered or inside
            inc = abs(x - (x1 + x2) / 2) <= rad and abs(y - (y1 + y2) / 2) <= rad
            mx = max(l, t, r, b)
            if inside and inc and lo <= mx < hi:
                area = max(x2 - x1, 0.0) * max(y2 - y1, 0.0)
                key = (area, gi)
                if best is None or key < best[0]:
                    best = (key, gi, (l, t, r, b))
        if best is None:
            labels.append(-2 if covered else -1)
            cen.append(0.0)
            reg.append([0.0, 0.0, 0.0, 0.0])
        else:
            _, gi, (l, t, r, b) = best
            labels.append(gi)
            fa = min(l, r) / max(l, r) if max(l, r) > 0 else 0.0
            fb = min(t, b) / max(t, b) if max(t, b) > 0 else 0.0
            cen.append(math.sqrt(fa * fb))
            reg.append([l, t, r, b])
    return labels, cen, reg


def _grid_points(sizes, strides, offsets=(0.5, 0.5)):
    pts, lv = [], []
    for li, ((h, w), s) in enumerate(zip(sizes, strides)):
        for y in range(h):
            for x in range(w):
                pts.append([(x + offsets[0]) * s, (y + offsets[1]) * s])
                lv.append(li)
    return pts, lv


def check_oracle(solve):
    """两层不同 stride、多个尺寸的 GT，逐点与纯 Python oracle 对拍。"""
    torch.manual_seed(11)
    configs = []
    pts, lv = _grid_points([(3, 3), (2, 2)], [8.0, 16.0])
    configs.append(([[6.0, 6.0, 40.0, 34.0], [40.0, 40.0, 60.0, 60.0]], pts, lv, [8.0, 16.0],
                    [(0.0, 48.0), (48.0, 200.0)]))
    pts2, lv2 = _grid_points([(4, 4)], [4.0], offsets=(0.5, 0.5))
    configs.append(([[10.0, 10.0, 26.0, 26.0]], pts2, lv2, [4.0], [(0.0, 100.0)]))
    # 大量小 GT，检验「最小面积优先」与 center 采样
    gts = [[i * 10.0, i * 10.0, i * 10.0 + 12.0, i * 10.0 + 12.0] for i in range(4)]
    configs.append((gts, pts2, lv2, [4.0], [(0.0, 100.0)]))
    for ci, (gt, points, levels, strides, ranges) in enumerate(configs):
        gt_t = torch.tensor(gt, dtype=torch.float32)
        pts_t = torch.tensor(points, dtype=torch.float32)
        got_l, got_c, got_r = solve(gt_t, pts_t, levels, strides, ranges)
        exp_l, exp_c, exp_r = _oracle(gt_t, points, levels, strides, ranges)
        assert got_l.tolist() == exp_l, (ci, got_l.tolist(), exp_l)
        assert torch.allclose(got_c, torch.tensor(exp_c), atol=1e-6), (ci, got_c.tolist(), exp_c)
        assert torch.allclose(got_r, torch.tensor(exp_r), atol=1e-6), (ci, got_r.tolist(), exp_r)
    # center_radius 变化也要跟着走
    gt_t = torch.tensor([[0.0, 0.0, 30.0, 30.0]], dtype=torch.float32)
    pts_t = torch.tensor([[2.0, 2.0], [15.0, 15.0], [28.0, 28.0]], dtype=torch.float32)
    for rad in (0.0, 0.25, 5.0):
        got = solve(gt_t, pts_t, [0, 0, 0], [8.0], [(0.0, 100.0)], center_radius=rad)
        exp = _oracle(gt_t, pts_t.tolist(), [0, 0, 0], [8.0], [(0.0, 100.0)], radius=rad)
        assert got[0].tolist() == exp[0], (rad, got[0].tolist(), exp[0])
        assert torch.allclose(got[1], torch.tensor(exp[1]), atol=1e-6), (rad, got[1])


def check_smallest_area(solve):
    """同一层两个嵌套 GT：取面积最小者，与 GT 顺序无关。"""
    big = [0.0, 0.0, 30.0, 30.0]   # area 900
    small = [8.0, 8.0, 16.0, 16.0]  # area 64
    pts = torch.tensor([[12.0, 12.0]], dtype=torch.float32)
    for gt_list, exp_label in (([big, small], 1), ([small, big], 0)):
        gt = torch.tensor(gt_list, dtype=torch.float32)
        labels, cen, reg = solve(gt, pts, [0], [8.0], [(0.0, 128.0)])
        assert labels.tolist() == [exp_label], (gt_list, labels.tolist())
        assert torch.allclose(cen, torch.tensor([1.0])), cen
        assert torch.allclose(reg, torch.tensor([[4.0, 4.0, 4.0, 4.0]])), reg
    # 面积相同 -> 取下标最小者
    same = [[4.0, 4.0, 20.0, 20.0], [4.0, 4.0, 20.0, 20.0]]
    labels, _, _ = solve(torch.tensor(same), pts, [0], [8.0], [(0.0, 128.0)])
    assert labels.tolist() == [0], labels.tolist()


def check_center_region_ignore(solve):
    """在框内但落在中心采样区外 -> 忽略样本 -2（不是负样本）。"""
    gt = torch.tensor([[0.0, 0.0, 30.0, 30.0]], dtype=torch.float32)
    pts = torch.tensor([[2.0, 2.0], [15.0, 15.0]], dtype=torch.float32)
    labels, cen, reg = solve(gt, pts, [0, 0], [8.0], [(0.0, 100.0)], center_radius=0.25)
    # radius = 2：只有 (15,15)（恰好格心）在中心区
    assert labels.tolist() == [-2, 0], labels.tolist()
    assert cen.tolist() == [0.0, 1.0], cen.tolist()
    assert reg.tolist() == [[0.0] * 4, [15.0, 15.0, 15.0, 15.0]], reg.tolist()
    # radius=0 只剩「恰好框心」一点
    labels0, _, _ = solve(gt, pts, [0, 0], [8.0], [(0.0, 100.0)], center_radius=0.0)
    assert labels0.tolist() == [-2, 0], labels0.tolist()


def check_regress_range_ignore(solve):
    """max(l,t,r,b) 超出层尺度区间 -> 忽略；区间左闭右开。"""
    gt = torch.tensor([[2.0, 2.0, 22.0, 22.0]], dtype=torch.float32)  # 20x20
    pts = torch.tensor([[4.0, 4.0], [12.0, 4.0], [12.0, 12.0]], dtype=torch.float32)
    # maxd: 18, 18, 10。区间 [0,16) 下前两点越界
    labels, cen, _ = solve(gt, pts, [0, 0, 0], [8.0], [(0.0, 16.0)])
    assert labels.tolist() == [-2, -2, 0], labels.tolist()
    assert cen.tolist() == [0.0, 0.0, 1.0], cen.tolist()
    # 区间 [0,18)：maxd 恰为 18 的两点落入右开端 -> 仍是忽略
    labels2, _, _ = solve(gt, pts, [0, 0, 0], [8.0], [(0.0, 18.0)])
    assert labels2.tolist() == [-2, -2, 0], labels2.tolist()
    # 区间 [0,20)：18 < 20，全部正样本
    labels3, cen3, _ = solve(gt, pts, [0, 0, 0], [8.0], [(0.0, 20.0)])
    assert labels3.tolist() == [0, 0, 0], labels3.tolist()
    # 左端闭：lo 恰等于 maxd 时仍算在区间内
    labels4, _, _ = solve(gt, pts, [0, 0, 0], [8.0], [(18.0, 40.0)])
    assert labels4.tolist() == [0, 0, -2], labels4.tolist()


def check_centerness_boundary(solve):
    """center-ness 的取值为 0：点落在框边或退化框上。"""
    gt = torch.tensor([[2.0, 2.0, 22.0, 22.0]], dtype=torch.float32)
    pts = torch.tensor([[2.0, 12.0], [22.0, 12.0], [12.0, 2.0], [12.0, 22.0],
                        [2.0, 2.0], [12.0, 12.0]], dtype=torch.float32)
    levels = [0] * 6
    labels, cen, reg = solve(gt, pts, levels, [16.0], [(0.0, 200.0)])
    assert labels.tolist() == [0] * 6, labels.tolist()
    assert cen.tolist() == [0.0, 0.0, 0.0, 0.0, 0.0, 1.0], cen.tolist()
    assert reg.tolist()[0] == [0.0, 10.0, 20.0, 10.0], reg.tolist()[0]
    # 零面积 GT：面积 0 是最小面积，与点重合时被选中，center-ness 按 0 处理
    deg = torch.tensor([[10.0, 10.0, 10.0, 10.0]], dtype=torch.float32)
    labels_d, cen_d, reg_d = solve(deg, torch.tensor([[10.0, 10.0]]), [0], [16.0],
                                   [(0.0, 100.0)])
    assert labels_d.tolist() == [0], labels_d.tolist()
    assert cen_d.tolist() == [0.0], cen_d.tolist()
    assert reg_d.tolist() == [[0.0, 0.0, 0.0, 0.0]], reg_d.tolist()


def check_no_gt_and_empty_points(solve):
    gt = torch.tensor([[2.0, 2.0, 22.0, 22.0]], dtype=torch.float32)
    empty_gt = torch.zeros(0, 4)
    pts = torch.tensor([[12.0, 12.0], [100.0, 100.0]], dtype=torch.float32)
    labels, cen, reg = solve(empty_gt, pts, [0, 0], [8.0], [(0.0, 64.0)])
    assert labels.tolist() == [-1, -1], labels.tolist()
    assert torch.equal(cen, torch.zeros(2)) and torch.equal(reg, torch.zeros(2, 4))
    # 空点集：三种输出都为空
    labels_e, cen_e, reg_e = solve(gt, torch.zeros(0, 2), [], [8.0], [(0.0, 64.0)])
    assert labels_e.shape == (0,) and cen_e.shape == (0,) and reg_e.shape == (0, 4)
    assert labels_e.dtype == torch.int64


def check_float64(solve):
    gt = torch.tensor(EXAMPLE_GT, dtype=torch.float64)
    pts = torch.tensor(EXAMPLE_POINTS, dtype=torch.float64)
    labels, cen, reg = solve(gt, pts, EXAMPLE_LEVELS, EXAMPLE_STRIDES, EXAMPLE_RANGES)
    assert cen.dtype == torch.float64 and reg.dtype == torch.float64
    assert labels.tolist() == EXAMPLE_LABELS
    assert torch.allclose(cen, torch.tensor(EXAMPLE_CENTERNESS, dtype=torch.float64), atol=1e-12)
    assert torch.allclose(reg, torch.tensor(EXAMPLE_REG, dtype=torch.float64), atol=1e-12)


def check_errors(solve):
    gt = torch.tensor([[2.0, 2.0, 22.0, 22.0]])
    pts = torch.tensor([[12.0, 12.0]])

    def bad(fn, desc):
        try:
            fn()
        except ValueError:
            return
        raise AssertionError(f"应抛 ValueError: {desc}")

    bad(lambda: solve(gt.to(torch.int64), pts, [0], [8.0], [(0.0, 64.0)]), "gt dtype")
    bad(lambda: solve(torch.zeros(2, 3), pts, [0], [8.0], [(0.0, 64.0)]), "gt 形状")
    bad(lambda: solve(gt, torch.zeros(2), [0], [8.0], [(0.0, 64.0)]), "points 形状")
    bad(lambda: solve(gt, pts.to(torch.float64), [0], [8.0], [(0.0, 64.0)]), "points dtype")
    bad(lambda: solve(gt, pts, [0, 0], [8.0], [(0.0, 64.0)]), "levels 长度")
    bad(lambda: solve(gt, pts, [1], [8.0], [(0.0, 64.0)]), "levels 越界")
    bad(lambda: solve(gt, pts, [-1], [8.0], [(0.0, 64.0)]), "levels 负")
    bad(lambda: solve(gt, pts, [0], [8.0], [(0.0, 64.0), (0.0, 8.0)]), "长度不一致")
    bad(lambda: solve(gt, pts, [0], [], []), "空层")
    bad(lambda: solve(gt, pts, [0], [0.0], [(0.0, 64.0)]), "stride 非正")
    bad(lambda: solve(gt, pts, [0], [8.0], [(64.0, 64.0)]), "lo == hi")
    bad(lambda: solve(gt, pts, [0], [8.0], [(9.0, 8.0)]), "lo > hi")
    bad(lambda: solve(gt, pts, [0], [8.0], [(0.0, 64.0)], center_radius=-1.0),
        "center_radius 负")
