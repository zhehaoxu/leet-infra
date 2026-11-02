import torch


def _oracle(scores, rois, k):
    """纯 Python 参照：逐 bin 取通道、bin 内算术平均、k² 个 bin 再等权平均。"""
    s = scores.tolist()
    c, _kk, h, w = scores.shape
    out_pooled, out_bins, out_sizes = [], [], []
    for roi in rois.tolist():
        x1 = int(min(max(roi[0], 0), w))
        y1 = int(min(max(roi[1], 0), h))
        x2 = int(min(max(roi[2], 0), w))
        y2 = int(min(max(roi[3], 0), h))
        rw, rh = max(x2 - x1, 0), max(y2 - y1, 0)
        bins, sizes = [], []
        for i in range(k):
            rs = y1 + (i * rh) // k
            re = y1 + ((i + 1) * rh) // k
            for j in range(k):
                cs = x1 + (j * rw) // k
                ce = x1 + ((j + 1) * rw) // k
                cnt = (re - rs) * (ce - cs)
                ch = i * k + j
                vals = []
                for ci in range(c):
                    if cnt == 0:
                        vals.append(0.0)
                    else:
                        acc = 0.0
                        for yy in range(rs, re):
                            for xx in range(cs, ce):
                                acc += s[ci][ch][yy][xx]
                        vals.append(acc / cnt)
                bins.append(vals)
                sizes.append(cnt)
        pooled = [sum(b[ci] for b in bins) / (k * k) for ci in range(c)]
        out_pooled.append(pooled)
        out_bins.append([[b[ci] for b in bins] for ci in range(c)])
        out_sizes.append(sizes)
    return out_pooled, out_bins, out_sizes


def check_example_single_bin(solve):
    # 1 个通道、k=1：整张图一个 bin，输出就是全图均值
    s = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=torch.float64)
    out = solve(s, torch.tensor([[0, 0, 2, 2]]), 1)
    assert torch.allclose(out["pooled"], torch.tensor([[2.5]], dtype=torch.float64)), out["pooled"]
    assert out["bin_sizes"].tolist() == [[4]]
    assert not bool(out["empty"].any())


def check_channel_per_bin(solve):
    # k=2：bin (i,j) 只读通道 i*2+j，四块拼起来是 [[0,1],[2,3]]
    s = torch.zeros(1, 4, 2, 2, dtype=torch.float64)
    for ch in range(4):
        s[0, ch] = float(ch)
    out = solve(s, torch.tensor([[0, 0, 2, 2]]), 2)
    assert out["bins"][0, 0].tolist() == [0.0, 1.0, 2.0, 3.0], out["bins"][0, 0].tolist()
    assert out["pooled"][0, 0].item() == 1.5, out["pooled"]


def check_uneven_bins(solve):
    # 宽 5、k=2：两列 bin 分别 2 列与 3 列，宽度不等但权重相同
    s = torch.zeros(1, 4, 2, 5, dtype=torch.float64)
    s[0, 0, :, :2] = 1.0
    s[0, 1, :, 2:] = 9.0
    s[0, 2, :, :2] = 1.0
    s[0, 3, :, 2:] = 9.0
    out = solve(s, torch.tensor([[0, 0, 5, 2]]), 2)
    assert out["bin_sizes"].tolist() == [[2, 3, 2, 3]], out["bin_sizes"]
    assert out["bins"][0, 0].tolist() == [1.0, 9.0, 1.0, 9.0], out["bins"]
    assert out["pooled"][0, 0].item() == 5.0, out["pooled"]


def check_bin_mean_not_global_mean(solve):
    # 1 个像素的 bin 与 3 个像素的 bin 等权：结果不是整幅图的均值
    s = torch.zeros(1, 4, 2, 3, dtype=torch.float64)
    s[0, 0, 0, 0] = 8.0
    out = solve(s, torch.tensor([[0, 0, 3, 2]]), 2)
    assert out["bin_sizes"].tolist() == [[1, 2, 1, 2]], out["bin_sizes"]
    assert out["bins"][0, 0].tolist() == [8.0, 0.0, 0.0, 0.0], out["bins"]
    assert out["pooled"][0, 0].item() == 2.0, out["pooled"]


def check_empty_bin_is_zero(solve):
    # 宽 1 却要 k=2：右列 bin 全空，按 0 计入 k² 平均（拉低结果）
    s = torch.ones(1, 4, 1, 1, dtype=torch.float64)
    out = solve(s, torch.tensor([[0, 0, 1, 1]]), 2)
    # 整数整除划分下，1x1 的 RoI 只有最后一个 bin 拿到像素，前三个是空 bin
    assert out["empty"].tolist() == [[True, True, True, False]], out["empty"]
    assert out["bin_sizes"].tolist() == [[0, 0, 0, 1]], out["bin_sizes"]
    assert out["bins"][0, 0].tolist() == [0.0, 0.0, 0.0, 1.0], out["bins"]
    assert out["pooled"][0, 0].item() == 0.25, out["pooled"]


def check_clipping(solve):
    s = torch.arange(16, dtype=torch.float64).reshape(1, 1, 4, 4)
    out = solve(s, torch.tensor([[-3, -3, 2, 2]]), 1)
    assert out["bounds"].tolist() == [[0, 0, 2, 2]], out["bounds"]
    assert out["pooled"][0, 0].item() == 2.5, out["pooled"]
    # 非对称 RoI：确认 bounds 是 [x1, y1, x2, y2] 而不是转置
    out3 = solve(s, torch.tensor([[1, 0, 3, 2]]), 1)
    assert out3["bounds"].tolist() == [[1, 0, 3, 2]], out3["bounds"]
    # 完全在图外：全部 bin 为空，输出 0
    out2 = solve(s, torch.tensor([[10, 10, 12, 12]]), 1)
    assert out2["bounds"].tolist() == [[4, 4, 4, 4]], out2["bounds"]
    assert out2["pooled"][0, 0].item() == 0.0
    assert bool(out2["empty"].all())


def check_degenerate_roi(solve):
    s = torch.ones(1, 4, 3, 3, dtype=torch.float64)
    out = solve(s, torch.tensor([[1, 1, 1, 1]]), 2)
    assert out["pooled"][0, 0].item() == 0.0, out["pooled"]
    assert bool(out["empty"].all())
    # 反向坐标（x2 < x1）也按零面积处理
    out2 = solve(s, torch.tensor([[2, 2, 1, 1]]), 2)
    assert out2["pooled"][0, 0].item() == 0.0, out2["pooled"]


def check_multi_class_independent(solve):
    # 两个类别共用同一组 bin 几何，但各读各的通道块
    s = torch.zeros(2, 4, 2, 2, dtype=torch.float64)
    s[0, 0] = 1.0
    s[1, 3] = 5.0
    out = solve(s, torch.tensor([[0, 0, 2, 2]]), 2)
    assert out["pooled"].tolist() == [[0.25, 1.25]], out["pooled"]
    assert out["bins"].shape == (1, 2, 4), out["bins"].shape


def check_validation(solve):
    s = torch.zeros(1, 4, 2, 2, dtype=torch.float64)
    r = torch.tensor([[0, 0, 2, 2]])
    bad = [
        (s, r, 0),
        (s, r, -1),
        (s, r, True),
        (s, r, 3),
        (torch.zeros(4, 2, 2, dtype=torch.float64), r, 2),
        (s, torch.tensor([[0.0, 0.0, 2.5, 2.0]]), 2),
        (s, torch.tensor([0, 0, 2, 2]), 2),
        (s, torch.tensor([[0, 0, 2]]), 2),
    ]
    for scores, rois, k in bad:
        try:
            solve(scores, rois, k)
        except ValueError:
            continue
        raise AssertionError(f"应拒绝 k={k}, rois={getattr(rois, 'shape', rois)}")
    # bool 是 int 子类：单独用通道数恰好匹配的输入验一次，避免被 kk 检查掩盖
    try:
        solve(torch.zeros(1, 1, 2, 2, dtype=torch.float64), r, True)
    except ValueError:
        pass
    else:
        raise AssertionError("k=True 应被拒绝")


def check_empty_rois(solve):
    s = torch.zeros(3, 4, 2, 2, dtype=torch.float64)
    out = solve(s, torch.zeros(0, 4, dtype=torch.float64), 2)
    assert tuple(out["pooled"].shape) == (0, 3), out["pooled"].shape
    assert tuple(out["bounds"].shape) == (0, 4)
    assert tuple(out["bin_sizes"].shape) == (0, 4)


def check_no_alias(solve):
    s = torch.ones(1, 1, 2, 2, dtype=torch.float64)
    r = torch.tensor([[0, 0, 2, 2]])
    out = solve(s, r, 1)
    assert s.tolist() == [[[[1.0, 1.0], [1.0, 1.0]]]], "入参不应被改"
    out["bins"][0, 0, 0] = 99.0
    assert out["pooled"][0, 0].item() == 1.0, "pooled 不应随 bins 原地改动"
    out["bounds"][0, 0] = 3
    assert r[0, 0].item() == 0, "bounds 不应与 rois 共享存储"


def check_against_oracle(solve):
    torch.manual_seed(20261105)
    for trial in range(40):
        k = int(torch.randint(1, 4, (1,)).item())
        c = int(torch.randint(1, 3, (1,)).item())
        h = int(torch.randint(1, 7, (1,)).item())
        w = int(torch.randint(1, 7, (1,)).item())
        s = torch.randint(-8, 9, (c, k * k, h, w), dtype=torch.int64).to(torch.float64)
        n = int(torch.randint(1, 4, (1,)).item())
        rois = []
        for _ in range(n):
            x1 = int(torch.randint(-2, w + 3, (1,)).item())
            y1 = int(torch.randint(-2, h + 3, (1,)).item())
            x2 = x1 + int(torch.randint(-1, w + 2, (1,)).item())
            y2 = y1 + int(torch.randint(-1, h + 2, (1,)).item())
            rois.append([x1, y1, x2, y2])
        rt = torch.tensor(rois, dtype=torch.float64)
        out = solve(s, rt, k)
        pooled, bins, sizes = _oracle(s, rt, k)
        assert torch.allclose(out["pooled"], torch.tensor(pooled, dtype=torch.float64), atol=1e-12), (trial, pooled)
        assert torch.allclose(out["bins"], torch.tensor(bins, dtype=torch.float64), atol=1e-12), trial
        assert out["bin_sizes"].tolist() == sizes, (trial, out["bin_sizes"].tolist(), sizes)


check_example_single_bin.example = True
check_channel_per_bin.example = True
