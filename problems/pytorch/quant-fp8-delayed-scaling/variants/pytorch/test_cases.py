import torch

FP32_MAX = float(torch.finfo(torch.float32).max)


def _ref(amax_seq, history_len=3, fp8_max=448.0, margin=0, algo="max", init_scale=1.0):
    """按官方 `_default_sf_compute` + `_update_amax_history` 的 torch.where 链重写一遍。"""
    hist = torch.zeros(history_len, dtype=torch.float32)
    scale = torch.tensor(float(init_scale), dtype=torch.float32)
    out = {"scales": [], "next": [], "amax": [], "hist": [], "kept": [], "clamped": []}
    for raw in amax_seq:
        hist[0] = float(raw)
        out["hist"].append(hist.clone().double())
        reduced = hist.max() if algo == "max" else hist[0].clone()
        out["scales"].append(scale.clone())
        out["amax"].append(reduced.clone().double())
        raw_sf = (torch.tensor(fp8_max, dtype=torch.float32) / reduced) / float(2 ** margin)
        sf = torch.where(reduced > 0.0, raw_sf, scale)
        sf = torch.where(torch.isfinite(reduced), sf, scale)
        kept = not (bool(reduced > 0.0) and bool(torch.isfinite(reduced)))
        clamped = bool(torch.isinf(sf))
        sf = torch.where(torch.isinf(sf), torch.full_like(sf, FP32_MAX), sf)
        out["next"].append(sf.clone())
        out["kept"].append(kept)
        out["clamped"].append(clamped)
        scale = sf
        if history_len > 1:
            hist = torch.roll(hist, -1, 0)
        hist[0] = 0.0
    out["final_scale"] = float(scale.item())
    out["final_hist"] = hist.double()
    return out


def check_delayed_one_step(solve):
    # 第一步用 init_scale，第二步才用第一步的 amax
    out = solve(torch.tensor([10.0, 20.0], dtype=torch.float64), history_len=2)
    assert out["scales"].tolist()[0] == 1.0, out["scales"]
    assert abs(out["scales"][1].item() - 44.8) < 1e-5, out["scales"]
    assert abs(out["next_scales"][1].item() - 22.4) < 1e-5, out["next_scales"]


def check_max_vs_most_recent(solve):
    seq = torch.tensor([10.0, 100.0, 1.0], dtype=torch.float64)
    a = solve(seq, history_len=3, algo="max")
    b = solve(seq, history_len=3, algo="most_recent")
    assert abs(a["next_scales"][-1].item() - 4.48) < 1e-5, a["next_scales"]
    assert abs(b["next_scales"][-1].item() - 448.0) < 1e-3, b["next_scales"]


def check_window_eviction(solve):
    # 窗口=2：连续两次 1.0 之后大 amax 才被挤出去
    out = solve(torch.tensor([100.0, 100.0, 1.0, 1.0], dtype=torch.float64), history_len=2)
    assert abs(out["next_scales"][2].item() - 4.48) < 1e-5, out["next_scales"][2]
    assert abs(out["next_scales"][3].item() - 448.0) < 1e-3, out["next_scales"][3]
    assert out["histories"][3].tolist() == [1.0, 1.0], out["histories"]


def check_history_roll_and_zero(solve):
    out = solve(torch.tensor([10.0, 20.0, 30.0], dtype=torch.float64), history_len=3)
    assert out["histories"].tolist() == [[10.0, 0.0, 0.0], [20.0, 0.0, 10.0], [30.0, 10.0, 20.0]]
    assert out["final_history"].tolist() == [0.0, 20.0, 30.0], out["final_history"]


def check_zero_amax_keeps_scale(solve):
    out = solve(torch.tensor([10.0, 0.0], dtype=torch.float64), history_len=1, algo="most_recent")
    assert abs(out["scales"][1].item() - 44.8) < 1e-5, out["scales"]
    assert abs(out["next_scales"][1].item() - 44.8) < 1e-5, out["next_scales"]
    assert out["kept"].tolist() == [False, True], out["kept"]
    assert not bool(out["clamped"].any())


def check_nonfinite_amax_keeps_scale(solve):
    inf = float("inf")
    out = solve(torch.tensor([2.0, inf, float("nan")], dtype=torch.float64), history_len=1, algo="most_recent")
    kept_scale = out["scales"][1].item()
    assert abs(kept_scale - 224.0) < 1e-3, out["scales"]
    assert abs(out["next_scales"][1].item() - kept_scale) < 1e-6, "inf 不应缩小 scale"
    assert abs(out["next_scales"][2].item() - kept_scale) < 1e-6, "nan 不应改 scale"
    assert out["kept"].tolist() == [False, True, True], out["kept"]


def check_tiny_amax_clamps_to_fp32_max(solve):
    out = solve(torch.tensor([1e-40], dtype=torch.float64), history_len=1)
    assert out["clamped"].tolist() == [True], out["clamped"]
    assert out["next_scales"][0].item() == FP32_MAX, out["next_scales"]


def check_margin(solve):
    out = solve(torch.tensor([7.0], dtype=torch.float64), history_len=1, margin=2)
    assert abs(out["next_scales"][0].item() - 16.0) < 1e-5, out["next_scales"]
    out2 = solve(torch.tensor([7.0], dtype=torch.float64), history_len=1, fp8_max=57344.0, margin=0)
    assert abs(out2["next_scales"][0].item() - 8192.0) < 1e-2, out2["next_scales"]


def check_float32_narrowing(solve):
    out = solve(torch.tensor([1234567.89], dtype=torch.float64), history_len=1)
    assert out["amax_used"][0].item() == 1234567.875, out["amax_used"]
    assert out["histories"][0][0].item() == 1234567.875, out["histories"]


def check_history_len_one(solve):
    # 长度 1：不滚动，归约结果恒等于本步 amax（max 与 most_recent 等价）
    a = solve(torch.tensor([3.0, 5.0], dtype=torch.float64), history_len=1, algo="max")
    b = solve(torch.tensor([3.0, 5.0], dtype=torch.float64), history_len=1, algo="most_recent")
    assert torch.equal(a["next_scales"], b["next_scales"])
    assert a["histories"].tolist() == [[3.0], [5.0]], a["histories"]


def check_validation(solve):
    seq = torch.tensor([1.0], dtype=torch.float64)
    bad = [
        dict(history_len=0),
        dict(history_len=-1),
        dict(history_len=True),
        dict(algo="median"),
        dict(margin=-1),
        dict(margin=1.0),
        dict(fp8_max=0.0),
        dict(fp8_max=-448.0),
        dict(init_scale=0.0),
        dict(init_scale=float("inf")),
    ]
    for kw in bad:
        args = dict(history_len=3)
        args.update(kw)
        try:
            solve(seq, **args)
        except ValueError:
            continue
        raise AssertionError(f"应拒绝 {kw}")


def check_empty_input(solve):
    out = solve(torch.tensor([], dtype=torch.float64), history_len=4)
    assert out["scales"].numel() == 0 and out["next_scales"].numel() == 0
    assert tuple(out["histories"].shape) == (0, 4)
    assert out["final_scale"] == 1.0
    assert out["final_history"].tolist() == [0.0, 0.0, 0.0, 0.0]


def check_no_alias(solve):
    seq = torch.tensor([1.0, 2.0], dtype=torch.float64)
    out = solve(seq, history_len=2)
    assert seq.tolist() == [1.0, 2.0], "入参不应被改"
    out["scales"][0] = 999.0
    assert out["next_scales"][0].item() < 999.0, "scales 与 next_scales 不应共享存储"


def check_against_source(solve):
    torch.manual_seed(20261104)
    for history_len in (1, 2, 3, 8):
        for algo in ("max", "most_recent"):
            for margin in (0, 1, 3):
                for trial in range(6):
                    n = int(torch.randint(0, 12, (1,)).item())
                    vals = []
                    for _ in range(n):
                        r = float(torch.rand(1).item())
                        if r < 0.12:
                            vals.append(0.0)
                        elif r < 0.2:
                            vals.append(float("inf"))
                        else:
                            vals.append(float(torch.rand(1).item() * 200.0 + 0.01))
                    ref = _ref(vals, history_len, 448.0, margin, algo, 1.0)
                    out = solve(torch.tensor(vals, dtype=torch.float64), history_len=history_len,
                                margin=margin, algo=algo)
                    empty = torch.zeros(0, dtype=torch.float64)
                    assert torch.equal(out["scales"], torch.stack(ref["scales"]).double() if ref["scales"] else empty)
                    assert torch.equal(out["next_scales"], torch.stack(ref["next"]).double() if ref["next"] else empty)
                    assert torch.equal(out["amax_used"], torch.tensor(ref["amax"], dtype=torch.float64))
                    assert out["kept"].tolist() == ref["kept"], (history_len, algo, margin, trial)
                    assert out["clamped"].tolist() == ref["clamped"], (history_len, algo, margin, trial)
                    assert out["final_scale"] == ref["final_scale"]
                    assert torch.equal(out["final_history"], ref["final_hist"])


check_delayed_one_step.example = True
check_zero_amax_keeps_scale.example = True
