"""LLM-Barber mask 重建步 -- 测试用例."""

import math

import numpy as np
import torch


def _mat(v, rows, cols):
    n = rows * cols
    base = torch.arange(n, dtype=torch.float32)
    x = ((base * (0.271 + v * 0.017)) % 1.0) * 2.0 - 1.0
    return x.reshape(rows, cols).contiguous()


def _scaler(v, rows, cols):
    """正的 grad² 滑动平均量（纹波无重复；√3 因子使 float32 舍入可观测）。"""
    n = rows * cols
    base = torch.arange(n, dtype=torch.float32)
    x = ((base * (0.61 + v * 0.013)) % 1.0) * 0.9 + 0.1
    return (x * x * 3.14159265358979).reshape(rows, cols).contiguous()


def _ref_mask(v, rows, cols):
    """稀疏参考权重的 0 位（~30% 被剪）。"""
    n = rows * cols
    base = torch.arange(n, dtype=torch.float32)
    keep = ((base * (0.37 + v * 0.011)) % 1.0) > 0.3
    return keep.reshape(rows, cols)


def _fixture(v=0.0, retain=True, **overrides):
    weights = {"q": _mat(v, 6, 5), "f": _mat(v + 3.0, 5, 4)}
    ref_weights, scalers = {}, {}
    for name, (r, c) in {"q": (6, 5), "f": (5, 4)}.items():
        mask = _ref_mask(v + (0.0 if name == "q" else 1.0), r, c)
        # 存活位取 0.9 倍 (模拟 SparseGPT 重构后的稀疏权重, 与稠密权重不同)
        ref_weights[name] = weights[name] * mask * 0.9
        scalers[name] = _scaler(v + (5.0 if name == "q" else 6.0), r, c)
    kw = dict(weights=weights, ref_weights=ref_weights, scalers=scalers,
              metric="WD", threshold=1.0, granularity="layer",
              retain_reconstruction=retain)
    kw.update(overrides)
    return kw


def _np_find_threshold(growth, prune, threshold):
    if growth.size == 0 or prune.size == 0:
        return float("inf"), float("-inf")
    gs = np.sort(growth)[::-1]
    ps = np.sort(prune)
    min_len = min(gs.size, ps.size)
    recons_num = int(np.sum((gs[:min_len] - ps[:min_len]) > 0))
    idx = int(recons_num * threshold)
    gt = gs[-1] if idx == gs.size else gs[idx]
    pt = ps[-1] if idx == ps.size else ps[idx]
    return float(gt), float(pt)


def _np_metric(comb, scaler, metric):
    if metric == "WD":
        return np.abs(comb) * np.sqrt(scaler)
    if metric == "W":
        return np.abs(comb)
    return np.sqrt(scaler)


def _oracle(weights, ref_weights, scalers, metric="WD", threshold=1.0,
            granularity="layer", retain_reconstruction=True):
    masks, metrics, combined = {}, {}, {}
    for name in weights:
        w = weights[name].numpy().astype(np.float64)
        rw = ref_weights[name].numpy().astype(np.float64)
        sc = scalers[name].numpy().astype(np.float64)
        mask = rw == 0
        comb = np.where(mask, w, rw) if retain_reconstruction else w.copy()
        combined[name] = comb
        masks[name] = mask
        metrics[name] = _np_metric(comb, sc, metric)

    block_gt = block_pt = None
    if granularity == "block":
        growth_cat = np.concatenate(
            [metrics[n][masks[n]] for n in weights])
        prune_cat = np.concatenate(
            [metrics[n][~masks[n]] for n in weights])
        block_gt, block_pt = _np_find_threshold(growth_cat, prune_cat, threshold)

    thresholds, rebuilt = {}, {}
    zero_cnt = total = 0
    for name in weights:
        m = metrics[name]
        mask = masks[name].copy()
        if granularity == "block":
            gt, pt = block_gt, block_pt
        elif granularity == "layer":
            gt, pt = _np_find_threshold(m[mask], m[~mask], threshold)
        elif granularity == "output1":
            gt = np.full(m.shape[0], float("inf"))
            pt = np.full(m.shape[0], float("-inf"))
            for j in range(m.shape[0]):
                gt[j], pt[j] = _np_find_threshold(m[j][mask[j]],
                                                  m[j][~mask[j]], threshold)
        else:
            gt = np.full(m.shape[1], float("inf"))
            pt = np.full(m.shape[1], float("-inf"))
            for j in range(m.shape[1]):
                gt[j], pt[j] = _np_find_threshold(m[:, j][mask[:, j]],
                                                  m[:, j][~mask[:, j]],
                                                  threshold)
        if granularity == "output1":
            growth_idx = m > gt.reshape(-1, 1)
            prune_idx = m < pt.reshape(-1, 1)
        elif granularity == "input1":
            growth_idx = m > gt.reshape(1, -1)
            prune_idx = m < pt.reshape(1, -1)
        else:
            growth_idx = m > gt
            prune_idx = m < pt
        mask[growth_idx] = False
        mask[prune_idx] = True
        masks[name] = mask
        thresholds[name] = (gt, pt)
        rebuilt[name] = np.where(mask, 0.0, combined[name])
        zero_cnt += int(mask.sum())
        total += mask.size
    return {"masks": masks, "rebuilt": rebuilt,
            "sparsity": zero_cnt / total, "thresholds": thresholds}


def _assert_close(a, e, atol=1e-9):
    assert a.shape == e.shape, (a.shape, e.shape)
    diff = (a.double() - e.double()).abs().max().item()
    assert diff <= atol, diff


def _expect_err(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except ValueError as e:
        return str(e)
    raise AssertionError("期望 ValueError 但未抛出")


def _assert_result(r, exp):
    assert set(r) == {"masks", "rebuilt", "sparsity", "thresholds"}
    for name in exp["masks"]:
        assert torch.equal(r["masks"][name],
                           torch.from_numpy(exp["masks"][name])), name
        _assert_close(r["rebuilt"][name], torch.from_numpy(exp["rebuilt"][name]))
        gt, pt = r["thresholds"][name]
        egt, ept = exp["thresholds"][name]
        if isinstance(egt, np.ndarray):
            assert np.allclose(gt, egt) and np.allclose(pt, ept), name
        else:
            assert gt == egt and pt == ept, (name, gt, egt, pt, ept)
    assert abs(r["sparsity"] - exp["sparsity"]) < 1e-12


# ---------------------------------------------------------------- 主体
def check_layer_oracle(solve):
    """默认 layer 粒度 + WD 度量: mask/重建权重/阈值/稀疏度全对拍."""
    kw = _fixture()
    r = solve(**kw)
    _assert_result(r, _oracle(**kw))
    # 交换确实发生: 存在恢复位 (ref==0 且现在存活且 rebuilt!=0) 与新剪位
    for name in kw["weights"]:
        revived = (~r["masks"][name]) & (kw["ref_weights"][name] == 0) \
            & (r["rebuilt"][name] != 0)
        newly_pruned = (kw["ref_weights"][name] != 0) & (r["rebuilt"][name] == 0)
        assert revived.any() or newly_pruned.any(), name
        assert r["rebuilt"][name][r["masks"][name]].abs().sum() == 0


def check_metric_variants(solve):
    """W 与 D 度量各自对拍; 且与 WD 结果不同 (scaler 非常量)."""
    for metric in ("W", "D"):
        kw = _fixture(metric=metric)
        r = solve(**kw)
        _assert_result(r, _oracle(**kw))
    kw_wd = _fixture()
    kw_w = dict(kw_wd, metric="W")
    kw_d = dict(kw_wd, metric="D")
    m_wd = solve(**kw_wd)
    m_w = solve(**kw_w)
    m_d = solve(**kw_d)
    diff_w = any(not torch.equal(m_wd["masks"][n], m_w["masks"][n])
                 for n in kw_wd["weights"])
    diff_d = any(not torch.equal(m_wd["masks"][n], m_d["masks"][n])
                 for n in kw_wd["weights"])
    assert diff_w and diff_d


def check_retain_false(solve):
    """retain_reconstruction=False: combined 整体取稠密权重 (经 metric 生效)."""
    kw = _fixture(retain=False)
    r = solve(**kw)
    _assert_result(r, _oracle(**kw))
    # retain 只影响被剪位的 combined (存活位 ref==w), rebuilt 可能相同;
    # 但被剪位的 metric 变了 -> 阈值应可见地不同
    r_t = solve(**_fixture(retain=True))
    assert any(r["thresholds"][n] != r_t["thresholds"][n]
               for n in kw["weights"])


def check_granularity_block(solve):
    """block: 两矩阵拼接后全局一对阈值 (各 name 阈值相同且来自拼接集合)."""
    kw = _fixture(granularity="block")
    r = solve(**kw)
    _assert_result(r, _oracle(**kw))
    gts = {n: r["thresholds"][n] for n in kw["weights"]}
    assert len({(v[0], v[1]) if isinstance(v[0], float) else "row"
                for v in gts.values()}) == 1
    # 与 layer 粒度阈值不同 (作用域不同)
    r_layer = solve(**_fixture(granularity="layer"))
    assert any(r["thresholds"][n] != r_layer["thresholds"][n]
               for n in kw["weights"])


def check_granularity_output1_input1(solve):
    """逐行/逐列阈值各自对拍; 且与 layer 结果不同."""
    for g in ("output1", "input1"):
        kw = _fixture(granularity=g)
        r = solve(**kw)
        _assert_result(r, _oracle(**kw))
        r_layer = solve(**dict(kw, granularity="layer"))
        assert any(not torch.equal(r["masks"][n], r_layer["masks"][n])
                   for n in kw["weights"]), g


def check_empty_sides(solve):
    """全剪/全存矩阵: 一侧为空 -> inf/-inf 阈值 -> mask 不动, rebuilt=0/combined."""
    weights = {"a": _mat(0.0, 4, 4)}
    ref = torch.zeros(4, 4)  # 全被剪 -> growth 侧为空
    scalers = {"a": _scaler(0.0, 4, 4)}
    r = solve(**dict(weights=weights, ref_weights={"a": ref},
                     scalers=scalers, metric="W", threshold=1.0,
                     granularity="layer", retain_reconstruction=True))
    gt, pt = r["thresholds"]["a"]
    assert gt == float("inf") and pt == float("-inf")
    assert torch.equal(r["masks"]["a"], torch.ones(4, 4, dtype=torch.bool))
    # mask 不动 -> rebuilt 全 0 (mask 位清零)
    assert r["rebuilt"]["a"].abs().sum() == 0
    # 全存 -> prune 侧为空, 同样不动, rebuilt == combined == 权重本身
    ref2 = weights["a"].clone()
    r2 = solve(**dict(weights=weights, ref_weights={"a": ref2},
                      scalers=scalers, metric="W", threshold=1.0,
                      granularity="layer", retain_reconstruction=True))
    gt2, pt2 = r2["thresholds"]["a"]
    assert gt2 == float("inf") and pt2 == float("-inf")
    assert torch.equal(r2["masks"]["a"], torch.zeros(4, 4, dtype=torch.bool))
    _assert_close(r2["rebuilt"]["a"], weights["a"])


def check_index_boundary_last_element(solve):
    """int(recons_num*threshold) == numel -> 阈值取该侧排序末元素."""
    # growth 侧 4 个高 metric (9), prune 侧 6 个低 metric (1)
    # -> recons_num=4 == growth 数 -> idx 越界, gt 取 growth_sort[-1]
    w = torch.tensor([[9.0, 9.0, 1.0, 1.0, 1.0],
                      [9.0, 9.0, 1.0, 1.0, 1.0]])
    ref = torch.tensor([[0.0, 0.0, 1.0, 1.0, 1.0],
                        [0.0, 0.0, 1.0, 1.0, 1.0]])
    kw = dict(weights={"a": w}, ref_weights={"a": ref},
              scalers={"a": torch.ones(2, 5)}, metric="W",
              threshold=1.0, granularity="layer",
              retain_reconstruction=False)
    r = solve(**kw)
    exp = _oracle(**kw)
    _assert_result(r, exp)
    gt, pt = r["thresholds"]["a"]
    assert gt == 9.0  # growth_sort[-1]: 越界保护分支
    assert pt == 1.0
    # m > 9 无位, m < 1 无位 -> mask 不动
    assert torch.equal(r["masks"]["a"], ref == 0)


def check_threshold_ratio_scales_swap(solve):
    """threshold 越小 -> 交换数单调不增 (0.0 不交换, 1.0 交换最多)."""
    kw0 = _fixture(threshold=0.0)
    kw5 = _fixture(threshold=0.5)
    kw1 = _fixture(threshold=1.0)
    r0, r5, r1 = solve(**kw0), solve(**kw5), solve(**kw1)
    for n in kw0["weights"]:
        swap0 = (r0["masks"][n] != (kw0["ref_weights"][n] == 0)).sum().item()
        swap5 = (r5["masks"][n] != (kw5["ref_weights"][n] == 0)).sum().item()
        swap1 = (r1["masks"][n] != (kw1["ref_weights"][n] == 0)).sum().item()
        assert swap0 <= swap5 <= swap1, n
    # threshold=0 -> idx=0 -> 阈值取排序首元素, growth(>) 与 prune(<) 均为空交换? 至少对拍
    _assert_result(r0, _oracle(**kw0))
    _assert_result(r5, _oracle(**kw5))


def check_strict_inequality(solve):
    """等值边界: growth 侧用 >、prune 侧用 < (等值不交换)."""
    w = torch.tensor([[5.0, 5.0, 3.0], [3.0, 3.0, 3.0]])
    ref = torch.tensor([[0.0, 5.0, 3.0], [3.0, 3.0, 3.0]])  # 仅 (0,0) 被剪
    kw = dict(weights={"a": w}, ref_weights={"a": ref},
              scalers={"a": torch.ones(2, 3)}, metric="W", threshold=1.0,
              granularity="layer", retain_reconstruction=False)
    r = solve(**kw)
    # growth=[5], prune=[5,3,3,3,3] -> recons=1, idx=1
    # -> gt=growth[-1]=5 (越界分支), pt=prune_sort[1]=3
    gt, pt = r["thresholds"]["a"]
    assert gt == 5.0 and pt == 3.0
    # m > 5 无位 (被剪位 5 不复活, > 而非 >=); m < 3 无位 (3 不被剪, < 而非 <=)
    assert bool(r["masks"]["a"][0, 0]) is True   # 保持被剪
    assert bool(r["masks"]["a"][0, 1]) is False  # 等值 5 不被剪
    assert r["masks"]["a"][1].tolist() == [False, False, False]
    assert r["rebuilt"]["a"][0, 0].item() == 0.0
    assert r["rebuilt"]["a"][0, 1].item() == 5.0


def check_pair_count_and_pt_guard(solve):
    """配对计数遇精确 0 差值不计数 (严格 >); idx==prune 侧 numel 时取末元素."""
    # (a) growth=[3,3], prune=[5,4,3]: 首对差 0 不计数 -> recons=0
    #     (若用 >= 则 recons=1, pt 从 3 变 4)
    w1 = torch.tensor([[3.0, 3.0, 5.0, 4.0, 3.0]])
    ref1 = torch.tensor([[0.0, 0.0, 5.0, 4.0, 3.0]])
    # (b) growth 3 个 9 全大于 prune 2 个 1 -> idx=2==prune 数 -> pt 取末元素
    w2 = torch.tensor([[9.0, 9.0, 9.0, 1.0, 1.0]])
    ref2 = torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0]])
    kw = dict(weights={"a": w1, "b": w2},
              ref_weights={"a": ref1, "b": ref2},
              scalers={"a": torch.ones(1, 5), "b": torch.ones(1, 5)},
              metric="W", threshold=1.0, granularity="layer",
              retain_reconstruction=False)
    r = solve(**kw)
    exp = _oracle(**kw)
    _assert_result(r, exp)
    gta, pta = r["thresholds"]["a"]
    assert gta == 3.0 and pta == 3.0
    gtb, ptb = r["thresholds"]["b"]
    assert gtb == 9.0 and ptb == 1.0
    # (a): m > 3 与 m < 3 均无位 -> mask 不动; (b) 同理
    assert torch.equal(r["masks"]["a"], ref1 == 0)
    assert torch.equal(r["masks"]["b"], ref2 == 0)


def check_purity(solve):
    """纯函数: 三个 dict 的张量均不被修改."""
    kw = _fixture()
    snap = {name: (kw["weights"][name].clone(), kw["ref_weights"][name].clone(),
                   kw["scalers"][name].clone()) for name in kw["weights"]}
    solve(**kw)
    for name in kw["weights"]:
        w0, r0, s0 = snap[name]
        assert torch.equal(kw["weights"][name], w0)
        assert torch.equal(kw["ref_weights"][name], r0)
        assert torch.equal(kw["scalers"][name], s0)
    # mask 是新张量 (不共享 ref 内存)
    r = solve(**kw)
    assert r["masks"]["q"].data_ptr() != (kw["ref_weights"]["q"] == 0).data_ptr()


def check_args_validation(solve):
    """dict/键/形状/非负 scaler/metric/granularity/threshold 校验."""
    kw = _fixture()
    assert "weights 必须是非空 dict" in _expect_err(solve, {}, kw["ref_weights"],
                                                  kw["scalers"])
    assert "weights[q] 必须是 2 维" in _expect_err(
        solve, {"q": torch.zeros(6)}, kw["ref_weights"], kw["scalers"])
    assert "weights[q] 必须是浮点张量" in _expect_err(
        solve, {"q": torch.zeros(6, 5, dtype=torch.long)},
        {"q": kw["ref_weights"]["q"]}, {"q": kw["scalers"]["q"]})
    bad = dict(kw, ref_weights={"q": kw["ref_weights"]["q"]})
    assert "键必须一致" in _expect_err(solve, **bad)
    bad = dict(kw, scalers={"q": kw["scalers"]["q"]})
    assert "键必须一致" in _expect_err(solve, **bad)
    bad = dict(kw, scalers={"q": kw["scalers"]["q"],
                            "f": torch.zeros(5, 3)})
    assert "scalers[f] 的形状" in _expect_err(solve, **bad)
    assert "键必须是非空字符串" in _expect_err(
        solve, {1: kw["weights"]["q"]}, {1: kw["ref_weights"]["q"]},
        {1: kw["scalers"]["q"]})
    bad = dict(kw, ref_weights={"q": kw["ref_weights"]["q"],
                                "f": torch.zeros(5, 5)})
    assert "ref_weights[f] 的形状" in _expect_err(solve, **bad)
    bad = dict(kw, scalers={"q": kw["scalers"]["q"],
                            "f": -kw["scalers"]["f"]})
    assert "必须非负" in _expect_err(solve, **bad)
    bad = dict(kw, metric="X")
    assert "metric 必须是" in _expect_err(solve, **bad)
    bad = dict(kw, granularity="row")
    assert "granularity 必须是" in _expect_err(solve, **bad)
    for tv in (1.5, -0.1, True, float("inf"), "0.5"):
        bad = dict(kw, threshold=tv)
        assert "threshold 必须是 [0, 1] 内的有限数值" in _expect_err(solve, **bad)
    bad = dict(kw, threshold=0)
    solve(**bad)  # int 0 合法


# ---------------------------------------------------------------- 元信息
check_layer_oracle.example = True
check_metric_variants.example = True
check_retain_false.example = True
check_granularity_block.example = True
check_granularity_output1_input1.example = True
check_empty_sides.example = True
check_index_boundary_last_element.example = True
check_threshold_ratio_scales_swap.example = True
check_strict_inequality.example = True
check_pair_count_and_pt_guard.example = True
check_purity.example = True
check_args_validation.example = True
