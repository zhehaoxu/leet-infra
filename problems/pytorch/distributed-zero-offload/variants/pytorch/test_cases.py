def _close(a, b, tol=1e-6):
    b = float(b)
    return abs(float(a) - b) <= tol * max(1.0, abs(b))


# 独立参考：逐项累加各分量再判定瓶颈
def _ref(N, bw, ct, pb, gb, ob):
    gpu = pb * N
    cpu = gb * N + ob * N
    pcie = gb * N + pb * N
    t = pcie / bw
    return float(gpu), float(cpu), float(pcie), (t > ct)


# 样例用例
def check_example(solve):
    gpu, cpu, pcie, boot = solve(1e9, 3.2e10, 0.1)
    assert _close(gpu, 2.0e9), gpu
    assert _close(cpu, 14.0e9), cpu
    assert _close(pcie, 4.0e9), pcie
    assert boot is True, boot


def check_memory_split_sum16(solve):
    # 默认字节下 gpu+cpu == 16·N（训练态总量，对齐 training-state-memory）
    for N in (1e9, 7.5e8, 1e6):
        gpu, cpu, _, _ = solve(N, 3.2e10, 1.0)
        assert _close(gpu + cpu, 16.0 * N), (N, gpu, cpu)
        assert gpu < 16.0 * N, "offload 后 GPU 显存应远小于全放 GPU 的 16N"


def check_bottleneck_strict_gt(solve):
    # t_pcie = 4e9/1e10 = 0.4；compute 恰好 0.4 → 严格大于为 False
    N, bw = 1e9, 1e10
    _, _, _, boot_eq = solve(N, bw, 0.4)
    assert boot_eq is False, boot_eq
    _, _, _, boot_less = solve(N, bw, 0.5)   # compute 0.5 > 0.4 → 非瓶颈
    assert boot_less is False, boot_less
    _, _, _, boot_more = solve(N, bw, 0.3)   # compute 0.3 < 0.4 → 瓶颈
    assert boot_more is True, boot_more


def check_faster_pcie_clears_bottleneck(solve):
    # 带宽越大 t_pcie 越小，瓶颈应消失
    _, _, _, slow = solve(1e9, 1e10, 0.1)     # t=0.4 > 0.1 瓶颈
    _, _, _, fast = solve(1e9, 1e11, 0.1)     # t=0.04 < 0.1 非瓶颈
    assert slow is True and fast is False, (slow, fast)


def check_custom_dtype_bytes(solve):
    # fp32 梯度/参数 + 只优化器 8（示意）时按自定义字节算
    gpu, cpu, pcie, _ = solve(1e6, 1e10, 1.0, param_bytes=4, grad_bytes=4, opt_bytes=8)
    assert _close(gpu, 4e6) and _close(cpu, 12e6) and _close(pcie, 8e6), (gpu, cpu, pcie)


def check_matches_ref(solve):
    for N, bw, ct, pb, gb, ob in [
        (1e9, 3.2e10, 0.1, 2, 2, 12),
        (5e8, 1.6e10, 0.2, 2, 2, 12),
        (2e9, 6e10, 0.05, 4, 4, 12),
        (1e7, 1e9, 0.0, 2, 2, 12),
    ]:
        got = solve(N, bw, ct, pb, gb, ob)
        ref = _ref(N, bw, ct, pb, gb, ob)
        assert len(got) == 4
        for g, r in zip(got[:3], ref[:3]):
            assert _close(g, r), (got, ref)
        assert bool(got[3]) == bool(ref[3]), (got, ref)


check_example.example = True
