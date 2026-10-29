import math

import torch

# 示例（实测回填）：单层 2x2、stride 8、scale 16、ratio 1.0、图像 32x32，GT=[4,4,20,20]
EXAMPLE_GT = [[4.0, 4.0, 20.0, 20.0]]
EXAMPLE_FEATURE_SIZES = [(2, 2)]
EXAMPLE_STRIDES = [8]
EXAMPLE_SCALES = [16.0]
EXAMPLE_RATIOS = [1.0]
EXAMPLE_IMAGE_SIZE = (32, 32)
EXAMPLE_ANCHORS = [[0.0, 0.0, 12.0, 12.0],
                   [4.0, 0.0, 20.0, 12.0],
                   [0.0, 4.0, 12.0, 20.0],
                   [4.0, 4.0, 20.0, 20.0]]
EXAMPLE_LABELS = [-1, -2, -2, 0]
EXAMPLE_COUNTS = [1, 1, 2]


def check_example(solve):
    gt = torch.tensor(EXAMPLE_GT, dtype=torch.float32)
    anchors, labels, counts = solve(gt, EXAMPLE_FEATURE_SIZES, EXAMPLE_STRIDES,
                                    EXAMPLE_SCALES, EXAMPLE_RATIOS, EXAMPLE_IMAGE_SIZE)
    assert anchors.shape == (4, 4) and anchors.dtype == torch.float32
    assert torch.allclose(anchors, torch.tensor(EXAMPLE_ANCHORS), atol=1e-6), anchors.tolist()
    assert torch.equal(labels, torch.tensor(EXAMPLE_LABELS, dtype=torch.int64)), labels.tolist()
    assert torch.equal(counts, torch.tensor(EXAMPLE_COUNTS, dtype=torch.int64)), counts.tolist()


check_example.example = True


def _iou_loop(a, b):
    aw = max(a[2] - a[0], 0.0)
    ah = max(a[3] - a[1], 0.0)
    bw = max(b[2] - b[0], 0.0)
    bh = max(b[3] - b[1], 0.0)
    ix = min(a[2], b[2]) - max(a[0], b[0])
    iy = min(a[3], b[3]) - max(a[1], b[1])
    inter = max(ix, 0.0) * max(iy, 0.0)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _oracle(gt, sizes, strides, scales, ratios, image_size, pos_thr=0.5, neg_thr=0.4):
    """纯 Python 逐层/逐格/逐组合生成 anchor，再逐 anchor 逐 GT 算 IoU 与标签。"""
    w_img, h_img = float(image_size[0]), float(image_size[1])
    anchors = []
    for (h, w), stride in zip(sizes, strides):
        for y in range(int(h)):
            for x in range(int(w)):
                cx = (x + 0.5) * float(stride)
                cy = (y + 0.5) * float(stride)
                for s in scales:
                    for r in ratios:
                        hw = 0.5 * float(s) * math.sqrt(float(r))
                        hh = 0.5 * float(s) / math.sqrt(float(r))
                        x1 = min(max(cx - hw, 0.0), w_img)
                        y1 = min(max(cy - hh, 0.0), h_img)
                        x2 = min(max(cx + hw, 0.0), w_img)
                        y2 = min(max(cy + hh, 0.0), h_img)
                        anchors.append([x1, y1, x2, y2])

    gl = gt.tolist()
    labels = [-1] * len(anchors)
    best = []
    for a in anchors:
        bi, bg = 0.0, -1
        for gi, g in enumerate(gl):
            v = _iou_loop(a, g)
            if v > bi:
                bi, bg = v, gi
        best.append((bi, bg))
    for k, (bi, bg) in enumerate(best):
        if bg < 0:
            continue
        if bi >= pos_thr:
            labels[k] = bg
        elif bi >= neg_thr:
            labels[k] = -2
        else:
            labels[k] = -1
    for gi in range(len(gl)):
        cand = [(best[k][0], k) for k in range(len(anchors)) if best[k][1] == gi]
        if not cand:
            continue
        cand.sort(key=lambda t: (-t[0], t[1]))
        labels[cand[0][1]] = gi
    pos = sum(1 for v in labels if v >= 0)
    neg = sum(1 for v in labels if v == -1)
    ign = sum(1 for v in labels if v == -2)
    return anchors, labels, [pos, neg, ign]


def check_pointer_oracle(solve):
    """独立逐 anchor 循环 oracle：anchor 枚举顺序、IoU 与标签口径全对齐。"""
    configs = [
        ([[2.0, 2.0, 30.0, 30.0]], [(2, 2)], [8], [16.0], [1.0], (32, 32)),
        ([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 40.0, 40.0]], [(3, 4)], [8], [8.0, 12.0],
         [0.5, 1.0, 2.0], (48, 40)),
        ([[5.0, 6.0, 18.0, 22.0]], [(2, 2), (1, 1)], [4, 8], [10.0], [1.0, 2.0], (40, 36)),
        ([[1.0, 1.0, 7.0, 9.0]], [(1, 1)], [4], [4.0], [1.0], (8, 8)),
    ]
    for ci, (gt, sizes, strides, scales, ratios, img) in enumerate(configs):
        gt_t = torch.tensor(gt, dtype=torch.float32)
        got_a, got_l, got_c = solve(gt_t, sizes, strides, scales, ratios, img)
        exp_a, exp_l, exp_c = _oracle(gt_t, sizes, strides, scales, ratios, img)
        assert got_a.shape == (len(exp_a), 4), (ci, got_a.shape)
        assert torch.allclose(got_a, torch.tensor(exp_a), atol=1e-4, rtol=1e-5), \
            (ci, (got_a - torch.tensor(exp_a)).abs().max().item())

        # 保底规则在「多个 anchor 与同一 GT 的 IoU 数学上完全并列」时，float32 与
        # float64 的末位噪声会分别选中不同 anchor；只承诺「每 GT 恰有一个正样本」，
        # 故并列候选位允许一侧为正、另一侧为负/忽略，其余位置必须逐位一致。
        gl = gt_t.tolist()
        iou = [[_iou_loop(a, g) for g in gl] for a in exp_a]
        gmax = [max((row[g] for row in iou), default=0.0) for g in range(len(gl))]
        ties = [[g for g in range(len(gl)) if iou[k][g] > 0.0 and iou[k][g] >= gmax[g] - 1e-6]
                for k in range(len(exp_l))]
        gl_list = got_l.tolist()
        for k, ev in enumerate(exp_l):
            gv = gl_list[k]
            if gv >= 0 or ev >= 0:
                assert (gv >= 0 and ev >= 0) or ties[k], (ci, k, gv, ev)
            else:
                assert gv == ev, (ci, k, gv, ev)
            if gv >= 0:
                assert iou[k][gv] >= gmax[gv] - 1e-6 or iou[k][gv] >= 0.5 - 1e-6, (ci, k, gv)
        for gi in range(gt_t.shape[0]):
            assert (got_l == gi).sum().item() >= 1, (ci, gi, gl_list)
        pos = int((got_l >= 0).sum())
        neg = int((got_l == -1).sum())
        ign = int((got_l == -2).sum())
        assert got_c.tolist() == [pos, neg, ign], (ci, got_c.tolist())
        assert pos == exp_c[0], (ci, pos, exp_c)


def check_anchor_enumeration(solve):
    """anchor 数 = Σ H*W*len(scales)*len(ratios)；顺序为 层→行→列→scale→ratio。"""
    gt = torch.tensor([[10.0, 10.0, 20.0, 20.0]])
    anchors, labels, counts = solve(gt, [(2, 3)], [8], [4.0, 8.0], [1.0], (64, 64))
    assert anchors.shape[0] == 2 * 3 * 2 * 1
    assert counts.sum().item() == anchors.shape[0]
    # 第一格中心 (4,4)、第二个 scale 的框更大（图像足够大，无裁剪）
    c0 = ((anchors[0, 0] + anchors[0, 2]) / 2, (anchors[0, 1] + anchors[0, 3]) / 2)
    assert abs(c0[0].item() - 4.0) < 1e-6 and abs(c0[1].item() - 4.0) < 1e-6, c0
    w0 = (anchors[0, 2] - anchors[0, 0]).item()
    w1 = (anchors[1, 2] - anchors[1, 0]).item()
    assert abs(w0 - 4.0) < 1e-6 and abs(w1 - 8.0) < 1e-6, (w0, w1)
    # 格优先顺序：前两个 anchor 同格，第三个已跳到下一格 (cx=12)
    c2 = ((anchors[2, 0] + anchors[2, 2]) / 2, (anchors[2, 1] + anchors[2, 3]) / 2)
    assert abs(c2[0].item() - 12.0) < 1e-6 and abs(c2[1].item() - 4.0) < 1e-6, c2
    # 多 ratio 时第二个 ratio 的宽高比
    a2, _, _ = solve(gt, [(1, 1)], [16], [8.0], [0.5, 2.0], (32, 32))
    w_a = (a2[0, 2] - a2[0, 0]).item()
    h_a = (a2[0, 3] - a2[0, 1]).item()
    w_b = (a2[1, 2] - a2[1, 0]).item()
    h_b = (a2[1, 3] - a2[1, 1]).item()
    assert abs(w_a / h_a - 0.5) < 1e-6 and abs(w_b / h_b - 2.0) < 1e-6


def check_threshold_boundaries(solve):
    """三档划分：IoU 1.0 为正、0.4706 落忽略带、0.1905 为负。"""
    gt = torch.tensor([[4.0, 4.0, 20.0, 20.0]], dtype=torch.float32)
    _, labels, counts = solve(gt, [(2, 2)], [8], [16.0], [1.0], (32, 32),
                              pos_thr=0.5, neg_thr=0.4)
    assert labels.tolist() == [-1, -2, -2, 0], labels.tolist()
    assert counts.tolist() == [1, 1, 2]
    # neg_thr 降到 0.2：0.1905 仍在负样本侧（< 0.2），忽略带不变
    _, labels2, counts2 = solve(gt, [(2, 2)], [8], [16.0], [1.0], (32, 32),
                                pos_thr=0.5, neg_thr=0.2)
    assert labels2.tolist() == [-1, -2, -2, 0], labels2.tolist()
    # neg_thr=pos_thr=0.4：0.4706 >= pos_thr 归正样本，忽略带消失
    _, labels3, counts3 = solve(gt, [(2, 2)], [8], [16.0], [1.0], (32, 32),
                                pos_thr=0.4, neg_thr=0.4)
    assert labels3.tolist() == [-1, 0, 0, 0], labels3.tolist()
    assert counts3.tolist() == [3, 1, 0]


def check_guarantee_positive(solve):
    """所有 IoU 都低于 pos_thr 时，每 GT 仍保底一个正样本（IoU 最大、并列取小下标）。"""
    gt = torch.tensor([[15.9, 15.9, 16.1, 16.1]], dtype=torch.float32)
    anchors, labels, counts = solve(gt, [(2, 2)], [8], [16.0], [1.0], (32, 32))
    assert labels.tolist() == [-1, -1, -1, 0], labels.tolist()
    assert counts.tolist() == [1, 3, 0]
    # 两个 GT 各自都要有正样本
    gt2 = torch.tensor([[1.0, 1.0, 2.0, 2.0], [30.0, 30.0, 31.5, 31.5]], dtype=torch.float32)
    _, labels2, counts2 = solve(gt2, [(4, 4)], [8], [16.0], [1.0], (32, 32), pos_thr=0.9)
    assert (labels2 == 0).sum().item() == 1 and (labels2 == 1).sum().item() == 1, labels2.tolist()


def check_no_gt(solve):
    anchors, labels, counts = solve(torch.zeros(0, 4), [(2, 3)], [8], [16.0], [1.0], (32, 32))
    assert anchors.shape == (6, 4)
    assert torch.equal(labels, torch.full((6,), -1, dtype=torch.int64))
    assert counts.tolist() == [0, 6, 0]
    # 空 GT 时 anchors 仍按层生成（与有 GT 时同形）
    a_gt, _, _ = solve(torch.tensor([[4.0, 4.0, 20.0, 20.0]]), [(2, 3)], [8], [16.0], [1.0], (32, 32))
    assert torch.equal(anchors, a_gt)


def check_clamp_to_image(solve):
    """anchor 越界部分被裁剪到图像边界内，裁剪后仍参与 IoU。"""
    gt = torch.tensor([[4.0, 4.0, 20.0, 20.0]], dtype=torch.float32)
    anchors, labels, counts = solve(gt, [(1, 1)], [16], [8.0], [1.0], (8, 8))
    assert anchors.tolist() == [[4.0, 4.0, 8.0, 8.0]], anchors.tolist()
    assert labels.tolist() == [0]  # 保底规则强制为正样本
    assert counts.tolist() == [1, 0, 0]
    # 所有 anchor 坐标都在 [0, W]x[0, H] 内
    a2, _, _ = solve(gt, [(3, 3)], [8], [24.0], [0.5, 2.0], (20, 20))
    assert (a2 >= 0).all() and (a2[:, [0, 2]] <= 20).all() and (a2[:, [1, 3]] <= 20).all()


def check_duplicate_gt_tie(solve):
    """重复 GT 时 best_gt 取下标最小者（并列），标签只写 0。"""
    gt = torch.tensor([[4.0, 4.0, 20.0, 20.0], [4.0, 4.0, 20.0, 20.0]], dtype=torch.float32)
    _, labels, counts = solve(gt, [(2, 2)], [8], [16.0], [1.0], (32, 32))
    assert labels.tolist() == [-1, -2, -2, 0], labels.tolist()
    assert counts.tolist() == [1, 1, 2]


def check_float64_and_dtype(solve):
    gt = torch.tensor([[4.0, 4.0, 20.0, 20.0]], dtype=torch.float64)
    anchors, labels, counts = solve(gt, [(2, 2)], [8], [16.0], [1.0], (32, 32))
    assert anchors.dtype == torch.float64
    assert labels.dtype == torch.int64 and counts.dtype == torch.int64
    assert torch.allclose(anchors, torch.tensor(EXAMPLE_ANCHORS, dtype=torch.float64), atol=1e-12)


def check_errors(solve):
    gt = torch.tensor([[4.0, 4.0, 20.0, 20.0]])

    def bad(fn, desc):
        try:
            fn()
        except ValueError:
            return
        raise AssertionError(f"应抛 ValueError: {desc}")

    bad(lambda: solve(gt.to(torch.int64), [(1, 1)], [8], [8.0], [1.0], (16, 16)), "dtype")
    bad(lambda: solve(torch.zeros(3), [(1, 1)], [8], [8.0], [1.0], (16, 16)), "gt 形状")
    bad(lambda: solve(gt, [], [], [8.0], [1.0], (16, 16)), "空层")
    bad(lambda: solve(gt, [(1, 1)], [8, 4], [8.0], [1.0], (16, 16)), "strides 长度")
    bad(lambda: solve(gt, [(0, 1)], [8], [8.0], [1.0], (16, 16)), "尺寸非正")
    bad(lambda: solve(gt, [(1, 1)], [0], [8.0], [1.0], (16, 16)), "stride 非正")
    bad(lambda: solve(gt, [(1, 1)], [8], [], [1.0], (16, 16)), "scales 空")
    bad(lambda: solve(gt, [(1, 1)], [8], [8.0], [0.0], (16, 16)), "ratio 非正")
    bad(lambda: solve(gt, [(1, 1)], [8], [8.0], [1.0], (0, 16)), "图像尺寸非正")
    bad(lambda: solve(gt, [(1, 1)], [8], [8.0], [1.0], (16, 16), pos_thr=0.3, neg_thr=0.5),
        "neg_thr > pos_thr")
