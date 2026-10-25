import torch


def _drive(solve, init_params, delta_seq, decay, warmup):
    """评测驱动：构造 EMA→每步先就地推进 params 再 update()。返回 (params, shadow)。"""
    ps = [p.clone() for p in init_params]
    ema = solve(ps, decay=decay, warmup=warmup)
    for deltas in delta_seq:
        for p, dl in zip(ps, deltas):
            p.data.add_(dl)
        ema.update()
    return ps, ema.shadow


def _oracle(init_params, delta_seq, decay, warmup):
    """独立重算：非就地、list 版，先推进当前参数再混合影子。"""
    cur = [p.clone() for p in init_params]
    shadow = [p.clone() for p in init_params]
    for n, deltas in enumerate(delta_seq, start=1):
        for c, dl in zip(cur, deltas):
            c += dl
        d = min(decay, (1 + n) / (10 + n)) if warmup else decay
        shadow = [d * s + (1 - d) * c for s, c in zip(shadow, cur)]
    return shadow


def check_example(solve):
    init = [torch.tensor([1.0, -2.0, 0.5])]
    inc = torch.tensor([0.1, -0.05, 0.2])
    delta_seq = [[inc] for _ in range(7)]
    _, shadow = _drive(solve, init, delta_seq, decay=0.9, warmup=True)
    exp = torch.tensor([1.6200042963027954, -2.310001850128174, 1.7400084733963013])
    assert torch.allclose(shadow[0], exp, atol=1e-5, rtol=1e-5), f"got {shadow[0].tolist()}"


check_example.example = True


def check_matches_oracle_multi(solve):
    init = [torch.tensor([1.0, -2.0, 0.5]), torch.tensor([[0.2, 0.4], [0.6, 0.8]])]
    delta_seq = [
        [torch.tensor([0.1, -0.05, 0.2]), torch.tensor([[0.03, -0.02], [0.05, 0.01]])],
        [torch.tensor([-0.2, 0.1, 0.05]), torch.tensor([[0.1, 0.0], [-0.05, 0.02]])],
        [torch.tensor([0.15, 0.2, -0.1]), torch.tensor([[-0.02, 0.07], [0.0, 0.03]])],
        [torch.tensor([0.05, -0.1, 0.3]), torch.tensor([[0.04, -0.06], [0.02, 0.0]])],
    ]
    _, shadow = _drive(solve, init, delta_seq, decay=0.9, warmup=True)
    exp = _oracle(init, delta_seq, 0.9, True)
    for a, b in zip(shadow, exp):
        assert torch.allclose(a, b, atol=1e-5, rtol=1e-5), f"偏离参考:\n{a}\nvs\n{b}"


def check_no_warmup_fixed_decay(solve):
    init = [torch.tensor([0.0, 0.0])]
    delta_seq = [[torch.tensor([1.0, 2.0])], [torch.tensor([1.0, 2.0])]]
    # decay=0.5 无升温：s1=0.5*0+0.5*[1,2]=[0.5,1]; s2=0.5*[0.5,1]+0.5*[2,4]=[1.25,2.5]
    _, shadow = _drive(solve, init, delta_seq, decay=0.5, warmup=False)
    exp = torch.tensor([1.25, 2.5])
    assert torch.allclose(shadow[0], exp, atol=1e-6, rtol=1e-6), f"got {shadow[0].tolist()}"
    ref = _oracle(init, delta_seq, 0.5, False)
    assert torch.allclose(shadow[0], ref[0], atol=1e-6, rtol=1e-6)


def check_decay_zero_tracks_exactly(solve):
    init = [torch.tensor([3.0, -1.0]), torch.tensor([0.5])]
    delta_seq = [[torch.tensor([0.7, 0.2]), torch.tensor([1.1])],
                 [torch.tensor([-0.3, 0.4]), torch.tensor([0.9])]]
    ps, shadow = _drive(solve, init, delta_seq, decay=0.0, warmup=False)
    # d=0 → 影子每次恰好等于当前 param
    for s, p in zip(shadow, ps):
        assert torch.allclose(s, p, atol=1e-7, rtol=1e-7), f"decay=0 应完全跟随: {s.tolist()} vs {p.tolist()}"


def check_shadow_independent_and_params_untouched(solve):
    params = [torch.tensor([1.0, 2.0]), torch.tensor([[3.0, 4.0], [5.0, 6.0]])]
    ema = solve(params, decay=0.9, warmup=True)
    assert all(sh is not p for sh, p in zip(ema.shadow, params)), "shadow 必须是与 param 分离的 clone"
    snap = [p.clone() for p in params]
    ema.update()
    ema.update()
    for p, s in zip(params, snap):
        assert torch.equal(p, s), f"update() 不该改动 params: {p.tolist()}"
