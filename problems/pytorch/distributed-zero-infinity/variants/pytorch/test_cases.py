def _close(a, b):
    return abs(a - b) <= 1e-6 * max(1.0, abs(b))


def _ref(N, d, op, oo, tgt, pb=2, gb=2, ob=12):
    """独立重算：先算全量训练态再减去被卸载分片（区别于逐分量相加的参考解）。"""
    p, g, o = pb * N / d, gb * N / d, ob * N / d
    offloaded = (p if op else 0.0) + (o if oo else 0.0)
    hbm = (p + g + o) - offloaded
    return {
        "hbm_bytes": hbm,
        "dram_bytes": offloaded if tgt == "dram" else 0.0,
        "ssd_bytes": offloaded if tgt == "ssd" else 0.0,
        "step_transfer_bytes": offloaded,
    }


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_example(solve):
    out = solve(7e9, 8, True, True, "ssd")
    assert _close(out["hbm_bytes"], 1.75e9), out
    assert _close(out["dram_bytes"], 0.0), out
    assert _close(out["ssd_bytes"], 1.225e10), out
    assert _close(out["step_transfer_bytes"], 1.225e10), out


def check_dram_target(solve):
    out = solve(7e9, 8, True, True, "dram")
    assert _close(out["dram_bytes"], 1.225e10) and _close(out["ssd_bytes"], 0.0), out
    assert _close(out["hbm_bytes"], 1.75e9), out


def check_no_offload_is_zero3(solve):
    # 两开关全 False → ZeRO-3 全驻 HBM = 16N/d，无搬运
    out = solve(7e9, 8, False, False, "ssd")
    assert _close(out["hbm_bytes"], 16 * 7e9 / 8), out
    assert _close(out["step_transfer_bytes"], 0.0), out
    assert _close(out["dram_bytes"], 0.0) and _close(out["ssd_bytes"], 0.0), out


def check_partition_scales_with_dp(solve):
    a = solve(7e9, 8, True, True, "ssd")
    b = solve(7e9, 16, True, True, "ssd")
    assert _close(b["hbm_bytes"], a["hbm_bytes"] / 2), "d 翻倍，每 rank HBM 减半"
    assert _close(b["step_transfer_bytes"], a["step_transfer_bytes"] / 2)


def check_matches_reference(solve):
    for (op, oo) in ((True, True), (False, True), (True, False), (False, False)):
        for tgt in ("dram", "ssd"):
            for d in (1, 4, 8, 16):
                out = solve(3e9, d, op, oo, tgt)
                ref = _ref(3e9, d, op, oo, tgt)
                for k in ref:
                    assert _close(out[k], ref[k]), (op, oo, tgt, d, k, out, ref)


check_example.example = True
