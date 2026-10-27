import torch

SPEC = torch.tensor([[0, 8, 4],
                     [1, 4, 0],
                     [2, 4, 6],
                     [0, 6, 6]], dtype=torch.long)


def _oracle(layer_spec):
    """独立重算：dict 分派逐层记账（结构异于参考解的分支累加）。"""
    table = {
        0: lambda a, b: (b, a * b + b),   # Linear(a->b, bias=True)：仅 bias 可训
        1: lambda a, b: (a, 2 * a),       # LayerNorm(a)：仅 β 可训，γ 不可训
        2: lambda a, b: (0, a * b),       # Linear(a->b, bias=False)
    }
    tr = tt = 0
    for row in layer_spec.tolist():
        kind, a, b = int(row[0]), int(row[1]), int(row[2])
        t_, o_ = table[kind](a, b)
        tr += t_
        tt += o_
    return int(tr), int(tt), float(tr) / float(tt)


def check_example(solve):
    tr, tt, ratio = solve(SPEC)
    assert tr == 14, f"n_trainable 应为 14，got {tr}"
    assert tt == 110, f"n_total 应为 110，got {tt}"
    assert abs(ratio - 14 / 110) <= 1e-9, f"ratio 应为 {14/110}，got {ratio}"


check_example.example = True


def check_matches_oracle(solve):
    torch.manual_seed(121)
    kinds = torch.randint(0, 3, (20,))
    a = torch.randint(1, 50, (20,))
    b = torch.randint(1, 50, (20,))
    b = torch.where(kinds == 1, torch.zeros_like(b), b)
    spec = torch.stack([kinds, a, b], dim=1)
    tr, tt, ratio = solve(spec)
    e_tr, e_tt, e_ratio = _oracle(spec)
    assert tr == e_tr and tt == e_tt, f"参数账应为 ({e_tr},{e_tt})，got ({tr},{tt})"
    assert abs(ratio - e_ratio) <= 1e-9, "ratio 与独立重算不符"


def check_ln_beta_only(solve):
    """全 LayerNorm 表：γ(weight) 不可训——可训 = Σa，总量 = 2Σa。"""
    spec = torch.tensor([[1, 5, 0], [1, 7, 0], [1, 3, 0]], dtype=torch.long)
    tr, tt, ratio = solve(spec)
    assert tr == 15, f"可训应只含 β（Σa=15），got {tr}"
    assert tt == 30, f"总量应为 2Σa=30，got {tt}"
    assert abs(ratio - 0.5) <= 1e-9, f"ratio 应为 0.5，got {ratio}"


def check_no_bias_linear(solve):
    """全 bias=False Linear：无 bias 无可训，ratio=0。"""
    spec = torch.tensor([[2, 3, 4], [2, 5, 2]], dtype=torch.long)
    tr, tt, ratio = solve(spec)
    assert tr == 0, f"可训应为 0，got {tr}"
    assert tt == 22, f"总量应为 12+10=22，got {tt}"
    assert ratio == 0.0, f"ratio 应为 0.0，got {ratio}"


def check_mixed_catalog(solve):
    """第二份手算混合表：Linear×3 + LayerNorm。"""
    spec = torch.tensor([[0, 3, 2], [1, 6, 0], [2, 10, 4], [0, 2, 2], [1, 8, 0]],
                        dtype=torch.long)
    tr, tt, ratio = solve(spec)
    # Linear(3->2): t=2,o=8；LN(6): t=6,o=12；Linear(10->4,nobias): t=0,o=40；
    # Linear(2->2): t=2,o=6；LN(8): t=8,o=16 → t=18, o=82
    assert tr == 18, f"n_trainable 应为 18，got {tr}"
    assert tt == 82, f"n_total 应为 82，got {tt}"
    assert abs(ratio - 18 / 82) <= 1e-9, f"ratio 应为 {18/82}，got {ratio}"
