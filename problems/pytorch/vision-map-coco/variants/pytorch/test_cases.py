import torch

# 示例（实测回填）：单类别 3 个 GT、4 个预测（1 个 FP 排在第 2 位）
EXAMPLE_PRED_BOXES = [[0.0, 0.0, 10.0, 10.0], [100.0, 100.0, 110.0, 110.0],
                      [20.0, 20.0, 30.0, 30.0], [40.0, 40.0, 50.0, 50.0]]
EXAMPLE_PRED_SCORES = [0.9, 0.8, 0.7, 0.6]
EXAMPLE_GT_BOXES = [[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0],
                    [40.0, 40.0, 50.0, 50.0]]
EXAMPLE_IOU_THRS = [0.5, 0.75]
EXAMPLE_AP_ALL = 0.8333333333333333
EXAMPLE_AP_11 = 0.8409090909090909
EXAMPLE_AP_COCO = 0.8341584158415841


def _tensors(boxes, scores, cls, img, gt_boxes, gt_cls, gt_img):
    return (torch.tensor(boxes, dtype=torch.float32), torch.tensor(scores, dtype=torch.float32),
            torch.tensor(cls, dtype=torch.int64), torch.tensor(img, dtype=torch.int64),
            torch.tensor(gt_boxes, dtype=torch.float32), torch.tensor(gt_cls, dtype=torch.int64),
            torch.tensor(gt_img, dtype=torch.int64))


def check_example(solve):
    args = _tensors(EXAMPLE_PRED_BOXES, EXAMPLE_PRED_SCORES, [0, 0, 0, 0], [0, 0, 0, 0],
                    EXAMPLE_GT_BOXES, [0, 0, 0], [0, 0, 0])
    for method, exp in (("all", EXAMPLE_AP_ALL), ("11point", EXAMPLE_AP_11),
                        ("coco", EXAMPLE_AP_COCO)):
        ap, mean_ap = solve(*args, iou_thrs=EXAMPLE_IOU_THRS, method=method)
        assert ap.shape == (2, 1) and ap.dtype == torch.float64, (method, ap.shape, ap.dtype)
        assert torch.allclose(ap, torch.full((2, 1), exp, dtype=torch.float64), atol=1e-12), \
            (method, ap.tolist())
        assert torch.allclose(mean_ap, torch.full((2,), exp, dtype=torch.float64), atol=1e-12), \
            (method, mean_ap.tolist())


check_example.example = True


def _iou(a, b):
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    inter = max(iw, 0.0) * max(ih, 0.0)
    aa = max(a[2] - a[0], 0.0) * max(a[3] - a[1], 0.0)
    ab = max(b[2] - b[0], 0.0) * max(b[3] - b[1], 0.0)
    u = aa + ab - inter
    return inter / u if u > 0 else 0.0


def _ap_curve(prec, rec, method):
    if not prec:
        return 0.0
    if method == "all":
        env = []
        best = 0.0
        for p in reversed(prec):
            best = max(best, p)
            env.append(best)
        env.reverse()
        ap, prev = 0.0, 0.0
        for p, r in zip(env, rec):
            ap += (r - prev) * p
            prev = r
        return ap
    grid = 101 if method == "coco" else 11
    rs = [k / (grid - 1) for k in range(grid)]
    total = 0.0
    for r in rs:
        cands = [p for p, rc in zip(prec, rec) if rc >= r]
        total += max(cands) if cands else 0.0
    return total / grid


def _oracle(pred_boxes, pred_scores, pred_cls, pred_img, gt_boxes, gt_cls, gt_img, thrs, method):
    pb = pred_boxes.tolist()
    sl = [float(v) for v in pred_scores.tolist()]
    pc, pi = [int(v) for v in pred_cls.tolist()], [int(v) for v in pred_img.tolist()]
    gb = gt_boxes.tolist()
    gc, gi = [int(v) for v in gt_cls.tolist()], [int(v) for v in gt_img.tolist()]
    classes = sorted(set(gc))
    order = sorted(range(len(pb)), key=lambda i: (-sl[i], i))
    out = []
    for thr in thrs:
        row = []
        for c in classes:
            gts = [g for g in range(len(gb)) if gc[g] == c]
            used = set()
            prec, rec = [], []
            tp = fp = 0
            for p in order:
                if pc[p] != c:
                    continue
                best_v, best_g = -1.0, None
                for g in gts:
                    if g in used or gi[g] != pi[p]:
                        continue
                    v = _iou(pb[p], gb[g])
                    if v > best_v:
                        best_v, best_g = v, g
                if best_g is not None and best_v >= thr:
                    used.add(best_g)
                    tp += 1
                else:
                    fp += 1
                prec.append(tp / (tp + fp))
                rec.append(tp / len(gts))
            row.append(_ap_curve(prec, rec, method))
        out.append(row)
    return out


def check_oracle(solve):
    """多类别、多图像、多阈值的确定性用例：与纯 Python oracle 全量对拍。"""
    torch.manual_seed(3)
    n_gt, n_pred = 10, 18
    gt = torch.rand(n_gt, 4) * 60.0
    gt[:, 2:] = gt[:, :2] + torch.rand(n_gt, 2) * 25.0 + 2.0
    gt_cls = torch.randint(0, 3, (n_gt,))
    gt_img = torch.randint(0, 2, (n_gt,))
    # 预测：一半是 GT 的抖动副本（制造 TP），一半随机（制造 FP）
    pred = gt[torch.randint(0, n_gt, (n_pred,))].clone()
    jitter = (torch.rand(n_pred, 4) - 0.5) * 8.0
    pred = (pred + jitter).clamp(min=0.0)
    pred[: n_pred // 2] = gt[torch.arange(n_pred // 2) % n_gt]  # 精确命中
    pred_cls = gt_cls[torch.arange(n_pred) % n_gt].clone()
    pred_img = gt_img[torch.arange(n_pred) % n_gt].clone()
    pred_scores = torch.rand(n_pred)
    thrs = [0.3, 0.5, 0.75]
    for method in ("all", "11point", "coco"):
        ap, mean_ap = solve(pred, pred_scores, pred_cls, pred_img, gt, gt_cls, gt_img,
                            iou_thrs=thrs, method=method)
        exp = _oracle(pred, pred_scores, pred_cls, pred_img, gt, gt_cls, gt_img, thrs, method)
        exp_t = torch.tensor(exp, dtype=torch.float64)
        assert ap.shape == exp_t.shape, (method, ap.shape, exp_t.shape)
        assert torch.allclose(ap, exp_t, atol=1e-9), \
            (method, (ap - exp_t).abs().max().item(), ap.tolist(), exp)
        assert torch.allclose(mean_ap, exp_t.mean(dim=1), atol=1e-9), (method, mean_ap.tolist())


def check_methods_differ(solve):
    """三种口径在同一 PR 曲线上的取值互不相同（实测常量）。"""
    args = _tensors(EXAMPLE_PRED_BOXES, EXAMPLE_PRED_SCORES, [0, 0, 0, 0], [0, 0, 0, 0],
                    EXAMPLE_GT_BOXES, [0, 0, 0], [0, 0, 0])
    vals = {}
    for method in ("all", "11point", "coco"):
        ap, _ = solve(*args, iou_thrs=[0.5], method=method)
        vals[method] = ap[0, 0].item()
    assert vals["all"] < vals["coco"] < vals["11point"], vals
    assert abs(vals["all"] - EXAMPLE_AP_ALL) < 1e-12
    assert abs(vals["coco"] - EXAMPLE_AP_COCO) < 1e-12
    assert abs(vals["11point"] - EXAMPLE_AP_11) < 1e-12


def check_threshold_boundary(solve):
    """IoU 恰好等于阈值算命中（>=）：0.75 在 thr=0.75 命中、在 thr=0.76 落空。"""
    pred = torch.tensor([[20.0, 20.0, 30.0, 27.5]])
    scores = torch.tensor([0.9])
    pcls = torch.tensor([0])
    pimg = torch.tensor([0])
    gt = torch.tensor([[20.0, 20.0, 30.0, 30.0]])
    gcls = torch.tensor([0])
    gimg = torch.tensor([0])
    assert abs(_iou(pred[0].tolist(), gt[0].tolist()) - 0.75) < 1e-12
    ap_hit, _ = solve(pred, scores, pcls, pimg, gt, gcls, gimg, iou_thrs=[0.75], method="all")
    assert abs(ap_hit[0, 0].item() - 1.0) < 1e-12, ap_hit.tolist()
    ap_miss, _ = solve(pred, scores, pcls, pimg, gt, gcls, gimg, iou_thrs=[0.76], method="all")
    assert abs(ap_miss[0, 0].item()) < 1e-12, ap_miss.tolist()


def check_greedy_rules(solve):
    """贪心匹配：GT 只用一次、取 IoU 最大者、图像/类别都必须相同。"""
    two_gt = torch.tensor([[0.0, 0.0, 10.0, 10.0], [50.0, 50.0, 60.0, 60.0]])
    gcls2 = torch.tensor([0, 0])
    gimg2 = torch.tensor([0, 0])
    # 两个完全相同的预测：第一个 TP（占用 GT0），第二个对 GT1 无交集 -> FP
    pred = torch.tensor([[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0]])
    scores = torch.tensor([0.9, 0.8])
    pcls = torch.tensor([0, 0])
    pimg = torch.tensor([0, 0])
    ap, _ = solve(pred, scores, pcls, pimg, two_gt, gcls2, gimg2, iou_thrs=[0.5],
                  method="all")
    # TP、FP -> precision 1 / 0.5, recall 0.5 -> AP = 0.5（不是 1.0，也不是 0）
    assert abs(ap[0, 0].item() - 0.5) < 1e-12, ap.tolist()
    # 同一坐标但图像 id 不同 -> 不能匹配，AP 为 0
    ap2, _ = solve(pred, scores, pcls, torch.tensor([1, 1]), two_gt, gcls2, gimg2,
                   iou_thrs=[0.5], method="all")
    assert abs(ap2[0, 0].item()) < 1e-12, ap2.tolist()
    # 同一坐标但类别不同 -> 也不能匹配（类别 1 无 GT，不参与平均）
    ap3, _ = solve(pred, scores, torch.tensor([1, 1]), pimg, two_gt, gcls2, gimg2,
                   iou_thrs=[0.5], method="all")
    assert abs(ap3[0, 0].item()) < 1e-12, ap3.tolist()
    # 一个预测同时与两个 GT 相交时取 IoU 最大者：与 [0,0,10,6] 的 0.9167 > 0.55
    nested = torch.tensor([[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 6.0]])
    pred4 = torch.tensor([[0.0, 0.0, 10.0, 5.5]])
    ap4, _ = solve(pred4, torch.tensor([0.5]), torch.tensor([0]), torch.tensor([0]),
                   nested, torch.tensor([0, 0]), torch.tensor([0, 0]),
                   iou_thrs=[0.5], method="all")
    # 匹配到小框后还剩 1 个 GT 未命中 -> recall 0.5, precision 1 -> AP 0.5
    assert abs(ap4[0, 0].item() - 0.5) < 1e-12, ap4.tolist()


def check_class_handling(solve):
    """类别集合取 GT 类别；无预测的类别 AP=0；无 GT 的预测类别不参与平均。"""
    gt = torch.tensor([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]])
    gcls = torch.tensor([0, 0])
    gimg = torch.tensor([0, 0])
    pred = torch.tensor([[0.0, 0.0, 10.0, 10.0], [50.0, 50.0, 60.0, 60.0]])
    scores = torch.tensor([0.9, 0.8])
    pcls = torch.tensor([0, 7])  # 类别 7 没有 GT
    pimg = torch.tensor([0, 0])
    ap, mean_ap = solve(pred, scores, pcls, pimg, gt, gcls, gimg, iou_thrs=[0.5],
                        method="all")
    assert ap.shape == (1, 1), ap.shape  # 只有类别 0
    assert abs(ap[0, 0].item() - 0.5) < 1e-12, ap.tolist()
    assert abs(mean_ap[0].item() - 0.5) < 1e-12, mean_ap.tolist()
    # 类别 0 一个预测都没有 -> AP 0
    ap2, mean2 = solve(pred, scores, torch.tensor([7, 7]), pimg, gt, gcls, gimg,
                       iou_thrs=[0.5], method="coco")
    assert ap2.shape == (1, 1) and abs(ap2[0, 0].item()) < 1e-12, ap2.tolist()
    assert abs(mean2[0].item()) < 1e-12, mean2.tolist()


def check_empty_inputs(solve):
    pred = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    scores = torch.tensor([0.9])
    pcls = torch.tensor([0])
    pimg = torch.tensor([0])
    # 没有 GT：类别数为 0，mean_ap 全 0
    ap, mean_ap = solve(pred, scores, pcls, pimg, torch.zeros(0, 4),
                        torch.zeros(0, dtype=torch.int64), torch.zeros(0, dtype=torch.int64),
                        iou_thrs=[0.5, 0.75], method="all")
    assert ap.shape == (2, 0) and mean_ap.shape == (2,)
    assert torch.equal(mean_ap, torch.zeros(2, dtype=torch.float64))
    # 没有预测：AP 全 0
    gt = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    ap2, mean2 = solve(torch.zeros(0, 4), torch.zeros(0), torch.zeros(0, dtype=torch.int64),
                       torch.zeros(0, dtype=torch.int64), gt, torch.tensor([0]),
                       torch.tensor([0]), iou_thrs=[0.5], method="coco")
    assert ap2.shape == (1, 1) and abs(mean2[0].item()) < 1e-12


def check_dtype(solve):
    args = _tensors(EXAMPLE_PRED_BOXES, EXAMPLE_PRED_SCORES, [0, 0, 0, 0], [0, 0, 0, 0],
                    EXAMPLE_GT_BOXES, [0, 0, 0], [0, 0, 0])
    ap, mean_ap = solve(*args, iou_thrs=[0.5], method="all")
    assert ap.dtype == torch.float64 and mean_ap.dtype == torch.float64
    # float64 输入与 float32 输入给出同一 AP（内部统一 float64 计算）
    a64 = tuple(t.to(torch.float64) if t.dtype == torch.float32 else t for t in args)
    ap64, _ = solve(*a64, iou_thrs=[0.5], method="all")
    assert torch.allclose(ap, ap64, atol=1e-12), (ap.tolist(), ap64.tolist())


def check_errors(solve):
    pb = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    ps = torch.tensor([0.9])
    pc = torch.tensor([0])
    pi = torch.tensor([0])
    gb = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    gc = torch.tensor([0])
    gi = torch.tensor([0])

    def bad(fn, desc):
        try:
            fn()
        except ValueError:
            return
        raise AssertionError(f"应抛 ValueError: {desc}")

    bad(lambda: solve(pb.to(torch.int64), ps, pc, pi, gb, gc, gi), "pred dtype")
    bad(lambda: solve(torch.zeros(3), ps, pc, pi, gb, gc, gi), "pred 形状")
    bad(lambda: solve(pb, ps.to(torch.float64), pc, pi, gb, gc, gi), "scores dtype")
    bad(lambda: solve(pb, torch.zeros(2), pc, pi, gb, gc, gi), "scores 长度")
    bad(lambda: solve(pb, ps, pc.to(torch.float32), pi, gb, gc, gi), "cls dtype")
    bad(lambda: solve(pb, ps, torch.zeros(2, dtype=torch.int64), pi, gb, gc, gi), "cls 长度")
    bad(lambda: solve(pb, ps, pc, pi, gb.to(torch.float64), gc, gi), "gt dtype 不一致")
    bad(lambda: solve(pb, ps, pc, pi, gb, gc, torch.zeros(2, dtype=torch.int64)), "img 长度")
    bad(lambda: solve(pb, ps, pc, pi, gb, gc, gi, iou_thrs=[]), "iou_thrs 空")
    bad(lambda: solve(pb, ps, pc, pi, gb, gc, gi, iou_thrs=[-0.1]), "iou_thrs 负")
    bad(lambda: solve(pb, ps, pc, pi, gb, gc, gi, iou_thrs=[1.5]), "iou_thrs 超 1")
    bad(lambda: solve(pb, ps, pc, pi, gb, gc, gi, method="voc2007"), "method 未知")
