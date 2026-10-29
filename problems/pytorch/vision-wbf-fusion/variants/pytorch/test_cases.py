import torch

# 示例（实测回填）：两条 A/B 融合、一条独立 C、一条另一类别的 D
EXAMPLE_BOXES = [[0.0, 0.0, 10.0, 10.0], [1.0, 0.0, 11.0, 10.0],
                 [20.0, 20.0, 30.0, 30.0], [0.0, 0.0, 10.0, 10.0]]
EXAMPLE_SCORES = [0.6, 0.6, 0.9, 0.5]
EXAMPLE_LABELS = [0, 0, 0, 1]
EXAMPLE_FUSED = [[20.0, 20.0, 30.0, 30.0], [0.5, 0.0, 10.5, 10.0], [0.0, 0.0, 10.0, 10.0]]
EXAMPLE_FUSED_SCORES = [0.9, 0.6, 0.5]
EXAMPLE_FUSED_LABELS = [0, 0, 1]


def check_example(solve):
    boxes = torch.tensor(EXAMPLE_BOXES)
    scores = torch.tensor(EXAMPLE_SCORES)
    labels = torch.tensor(EXAMPLE_LABELS)
    fb, fs, fl = solve(boxes, scores, labels)
    assert fb.dtype == torch.float32 and fs.dtype == torch.float32 and fl.dtype == torch.int64
    assert fb.shape == (3, 4) and fs.shape == (3,) and fl.shape == (3,)
    assert torch.allclose(fb, torch.tensor(EXAMPLE_FUSED), atol=1e-6), fb.tolist()
    assert torch.allclose(fs, torch.tensor(EXAMPLE_FUSED_SCORES), atol=1e-6), fs.tolist()
    assert torch.equal(fl, torch.tensor(EXAMPLE_FUSED_LABELS)), fl.tolist()


check_example.example = True


def _iou(a, b):
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    inter = max(iw, 0.0) * max(ih, 0.0)
    aa = max(a[2] - a[0], 0.0) * max(a[3] - a[1], 0.0)
    ab = max(b[2] - b[0], 0.0) * max(b[3] - b[1], 0.0)
    u = aa + ab - inter
    return inter / u if u > 0 else 0.0


def _oracle(boxes, scores, labels, iou_thr=0.55, num_models=1):
    """纯 Python：显式遍历排序序列、逐簇求加权和，不依赖张量。"""
    bl = boxes.tolist()
    sl = [float(v) for v in scores.tolist()]
    ll = [int(v) for v in labels.tolist()]
    order = sorted(range(len(bl)), key=lambda i: (-sl[i], i))
    clusters = []  # [label, member_indices, fused_box]
    for i in order:
        best_j, best_iou = -1, iou_thr
        for j, c in enumerate(clusters):
            if c[0] != ll[i]:
                continue
            v = _iou(bl[i], c[2])
            if v > best_iou:
                best_iou, best_j = v, j
        if best_j >= 0:
            c = clusters[best_j]
            c[1].append(i)
            wsum = sum(sl[k] for k in c[1])
            if wsum > 0:
                c[2] = [sum(bl[k][d] * sl[k] for k in c[1]) / wsum for d in range(4)]
            else:
                c[2] = [sum(bl[k][d] for k in c[1]) / len(c[1]) for d in range(4)]
        else:
            clusters.append([ll[i], [i], list(bl[i])])
    out = []
    for ci, (lab, mem, box) in enumerate(clusters):
        n = len(mem)
        sc = (sum(sl[k] for k in mem) / n) * min(n, num_models) / num_models
        out.append((ci, -sc, box, sc, lab))
    out.sort(key=lambda t: (t[1], t[0]))
    return [t[2] for t in out], [t[3] for t in out], [t[4] for t in out]


def check_oracle(solve):
    """确定性随机框集：与纯 Python oracle 逐项对拍（含多类别、多尺度重叠）。"""
    torch.manual_seed(7)
    configs = []
    boxes = torch.rand(24, 4) * 100.0
    boxes[:, 2:] = boxes[:, :2] + torch.rand(24, 2) * 30.0 + 1.0
    scores = torch.rand(24)
    labels = torch.randint(0, 3, (24,))
    configs.append((boxes, scores, labels, 0.55, 1))
    # 大量重叠：同一位置抖动，应融合成一簇
    base = torch.tensor([10.0, 10.0, 50.0, 50.0])
    boxes2 = base.repeat(6, 1) + (torch.arange(6).float().unsqueeze(1) * 1.5)
    configs.append((boxes2, torch.rand(6) * 0.5 + 0.5, torch.zeros(6, dtype=torch.int64),
                    0.6, 1))
    # num_models 缩放
    configs.append((boxes.clone(), scores.clone(), labels.clone(), 0.5, 3))
    configs.append((boxes2.clone(), torch.rand(6), torch.zeros(6, dtype=torch.int64), 0.3, 2))
    for ci, (b, s, l, thr, nm) in enumerate(configs):
        got = solve(b, s, l, iou_thr=thr, num_models=nm)
        exp = _oracle(b, s, l, iou_thr=thr, num_models=nm)
        assert got[0].shape == (len(exp[0]), 4), (ci, got[0].shape, len(exp[0]))
        assert torch.allclose(got[0], torch.tensor(exp[0]), atol=1e-5), \
            (ci, (got[0] - torch.tensor(exp[0])).abs().max().item())
        assert torch.allclose(got[1], torch.tensor(exp[1]), atol=1e-6), \
            (ci, got[1].tolist(), exp[1])
        assert got[2].tolist() == exp[2], (ci, got[2].tolist(), exp[2])


def check_model_rescale(solve):
    """num_models 缩放：单框簇按 1/T 降权，多框簇按 min(n,T)/T，会改变输出顺序。"""
    boxes = torch.tensor(EXAMPLE_BOXES)
    scores = torch.tensor(EXAMPLE_SCORES)
    labels = torch.tensor(EXAMPLE_LABELS)
    fb, fs, fl = solve(boxes, scores, labels, num_models=2)
    # 簇大小 1/2/1 -> 系数 0.5/1.0/0.5
    assert torch.allclose(fs, torch.tensor([0.6, 0.45, 0.25]), atol=1e-6), fs.tolist()
    assert torch.allclose(fb, torch.tensor([[0.5, 0.0, 10.5, 10.0], [20.0, 20.0, 30.0, 30.0],
                                            [0.0, 0.0, 10.0, 10.0]]), atol=1e-6), fb.tolist()
    assert fl.tolist() == [0, 0, 1], fl.tolist()
    # num_models=1 时缩放因子恒为 min(n,1)/1 = 1，与原分数一致
    _, fs1, _ = solve(boxes, scores, labels, num_models=1)
    assert torch.allclose(fs1, torch.tensor([0.9, 0.6, 0.5]), atol=1e-6), fs1.tolist()
    # num_models 大于簇大小时 min(n,T)/T = n/T：0.9*1/4, 0.6*2/4, 0.5*1/4
    _, fs3, _ = solve(boxes, scores, labels, num_models=4)
    assert torch.allclose(fs3, torch.tensor([0.3, 0.225, 0.125]), atol=1e-6), fs3.tolist()


def check_threshold_strict(solve):
    """合并判据是 IoU 严格大于阈值：IoU 恰为 0.5 时 thr=0.5 不合并、thr=0.49 合并。"""
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 5.0]])
    scores = torch.tensor([0.9, 0.8])
    labels = torch.tensor([0, 0])
    assert abs(_iou(boxes[0].tolist(), boxes[1].tolist()) - 0.5) < 1e-12
    fb, fs, _ = solve(boxes, scores, labels, iou_thr=0.5)
    assert fb.shape[0] == 2, fb.tolist()
    assert torch.allclose(fs, torch.tensor([0.9, 0.8])), fs.tolist()
    fb2, fs2, _ = solve(boxes, scores, labels, iou_thr=0.49)
    assert fb2.shape[0] == 1, fb2.tolist()
    # 加权平均（0.9/0.8）：y2 = (10*0.9 + 5*0.8)/1.7 = 7.6470588...
    assert torch.allclose(fb2, torch.tensor([[0.0, 0.0, 10.0, 7.6470588]]), atol=1e-5), \
        fb2.tolist()
    assert torch.allclose(fs2, torch.tensor([0.85])), fs2.tolist()


def check_label_separation(solve):
    """完全相同的框但类别不同：不合并，各自成簇。"""
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0],
                          [0.0, 0.0, 10.0, 10.0]])
    # 0.75/0.25 的均值恰为 0.5，与类别 1 的单框 0.5 完全同分，检验并列按创建顺序
    scores = torch.tensor([0.75, 0.5, 0.25])
    labels = torch.tensor([0, 1, 0])
    fb, fs, fl = solve(boxes, scores, labels)
    assert fb.shape == (2, 4), fb.tolist()
    assert torch.allclose(fs, torch.tensor([0.5, 0.5])), fs.tolist()
    assert fl.tolist() == [0, 1], fl.tolist()


def check_weighted_fusion(solve):
    """融合框是置信度加权平均，不是等权平均。"""
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [2.0, 2.0, 12.0, 12.0]])
    scores = torch.tensor([0.75, 0.25])
    labels = torch.tensor([0, 0])
    fb, fs, _ = solve(boxes, scores, labels, iou_thr=0.4)
    assert fb.shape[0] == 1, fb.tolist()
    assert torch.allclose(fb, torch.tensor([[0.5, 0.5, 10.5, 10.5]]), atol=1e-6), fb.tolist()
    # 等权平均会得到 [1,1,11,11]，权重取反则是 [1.5,1.5,11.5,11.5]
    assert not torch.allclose(fb, torch.tensor([[1.0, 1.0, 11.0, 11.0]]))
    assert torch.allclose(fs, torch.tensor([0.5])), fs.tolist()
    # 权重全 0 -> 退化为等权平均
    fb0, fs0, _ = solve(boxes, torch.zeros(2), labels, iou_thr=0.4, num_models=2)
    assert torch.allclose(fb0, torch.tensor([[1.0, 1.0, 11.0, 11.0]]), atol=1e-6), fb0.tolist()
    assert torch.allclose(fs0, torch.tensor([0.0])), fs0.tolist()


def check_best_iou_cluster(solve):
    """一个框同时命中多个簇时，并入 IoU 最大的簇（不是创建顺序第一个）。"""
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0],   # A -> 簇0
                          [12.0, 0.0, 22.0, 10.0],  # B -> 簇1
                          [1.0, 0.0, 22.0, 10.0]])  # X 同时命中两簇
    scores = torch.tensor([0.9, 0.8, 0.5])
    labels = torch.tensor([0, 0, 0])
    # IoU(X, A) = 90/220 ≈ 0.4091, IoU(X, B) = 100/210 ≈ 0.4762，两者都 > 0.4
    assert _iou(boxes[2].tolist(), boxes[0].tolist()) > 0.4
    assert _iou(boxes[2].tolist(), boxes[1].tolist()) > 0.4
    assert _iou(boxes[2].tolist(), boxes[1].tolist()) > _iou(boxes[2].tolist(), boxes[0].tolist())
    fb, fs, _ = solve(boxes, scores, labels, iou_thr=0.4)
    assert fb.shape[0] == 2, fb.tolist()
    # 簇0 只剩 A；簇1 = B 与 X 的加权平均
    assert torch.allclose(fb, torch.tensor([[0.0, 0.0, 10.0, 10.0],
                                            [10.1 / 1.3, 0.0, 22.0, 10.0]]), atol=1e-5), \
        fb.tolist()
    assert torch.allclose(fs, torch.tensor([0.9, 0.65]), atol=1e-6), fs.tolist()


def check_order_and_ties(solve):
    """处理顺序按置信度降序、并列取原下标小者；输出按融合分数降序、并列按创建顺序。"""
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [50.0, 50.0, 60.0, 60.0],
                          [100.0, 100.0, 110.0, 110.0]])
    scores = torch.tensor([0.5, 0.7, 0.5])
    labels = torch.tensor([0, 0, 0])
    fb, fs, _ = solve(boxes, scores, labels)
    assert torch.allclose(fs, torch.tensor([0.7, 0.5, 0.5])), fs.tolist()
    # 两个 0.5 的簇创建顺序：下标 0 先于下标 2
    assert fb[1, 0].item() == 0.0 and fb[2, 0].item() == 100.0, fb.tolist()


def check_empty_and_single(solve):
    fb, fs, fl = solve(torch.zeros(0, 4), torch.zeros(0), torch.zeros(0, dtype=torch.int64))
    assert fb.shape == (0, 4) and fs.shape == (0,) and fl.shape == (0,)
    fb1, fs1, fl1 = solve(torch.tensor([[1.0, 2.0, 3.0, 4.0]]), torch.tensor([0.42]),
                          torch.tensor([9]))
    assert fb1.tolist() == [[1.0, 2.0, 3.0, 4.0]], fb1.tolist()
    assert torch.allclose(fs1, torch.tensor([0.42])), fs1.tolist()
    assert fl1.tolist() == [9], fl1.tolist()


def check_float64(solve):
    boxes = torch.tensor(EXAMPLE_BOXES, dtype=torch.float64)
    scores = torch.tensor(EXAMPLE_SCORES, dtype=torch.float64)
    labels = torch.tensor(EXAMPLE_LABELS)
    fb, fs, fl = solve(boxes, scores, labels)
    assert fb.dtype == torch.float64 and fs.dtype == torch.float64
    assert torch.allclose(fb, torch.tensor(EXAMPLE_FUSED, dtype=torch.float64), atol=1e-12)
    assert torch.allclose(fs, torch.tensor(EXAMPLE_FUSED_SCORES, dtype=torch.float64),
                          atol=1e-12)
    assert fl.tolist() == EXAMPLE_FUSED_LABELS


def check_errors(solve):
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    scores = torch.tensor([0.5])
    labels = torch.tensor([0])

    def bad(fn, desc):
        try:
            fn()
        except ValueError:
            return
        raise AssertionError(f"应抛 ValueError: {desc}")

    bad(lambda: solve(boxes.to(torch.int64), scores, labels), "boxes dtype")
    bad(lambda: solve(torch.zeros(4), scores, labels), "boxes 形状")
    bad(lambda: solve(boxes, scores.to(torch.float64), labels), "scores dtype")
    bad(lambda: solve(boxes, torch.zeros(2), labels), "scores 形状")
    bad(lambda: solve(boxes, scores, labels.to(torch.float32)), "labels dtype")
    bad(lambda: solve(boxes, scores, torch.zeros(2, dtype=torch.int64)), "labels 形状")
    bad(lambda: solve(boxes, scores, labels, iou_thr=-0.1), "iou_thr 负")
    bad(lambda: solve(boxes, scores, labels, iou_thr=1.5), "iou_thr 超 1")
    bad(lambda: solve(boxes, scores, labels, num_models=0), "num_models=0")
    bad(lambda: solve(boxes, scores, labels, num_models=2.5), "num_models 非整数")
