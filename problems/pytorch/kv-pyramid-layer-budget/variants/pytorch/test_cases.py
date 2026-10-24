import torch


def _oracle(q_len, num_layers, cr, ws, beta):
    """独立：分数运算用 Fraction，最后一步手动 floor((n + 0.5)) 避开 banker's。"""
    from fractions import Fraction
    mcp = Fraction(ws) + Fraction(q_len) * (1 - Fraction(str(cr)))
    mn = (mcp - ws) / Fraction(beta)
    mx = (mcp - ws) * 2 - mn
    if mx >= q_len - ws:
        mx = Fraction(q_len - ws)
        mn = (mcp - ws) * 2 - mx
    if not (q_len >= mx >= mn >= ws):
        v = Fraction(q_len) * (1 - Fraction(str(cr)))
        # 使用与 round(x) 相同语义（banker's）——这里 Python round 直接调
        r = max(1, round(float(v)))
        return torch.tensor([r] * num_layers, dtype=torch.long)
    if num_layers > 1:
        steps = (mx - mn) / Fraction(num_layers - 1)
    else:
        steps = Fraction(0)
    out = []
    for l in range(num_layers):
        val = float(mx - l * steps)
        out.append(round(val))
    return torch.tensor(out, dtype=torch.long)


def check_example(solve):
    out = solve(100, 4, 0.5, 4, 2)
    exp = torch.tensor([75, 58, 42, 25])
    assert torch.equal(out, exp), out


def check_matches_oracle(solve):
    for q_len in (64, 100, 128, 256):
        for num_layers in (1, 2, 3, 4, 8, 12):
            for cr in (0.0, 0.25, 0.5, 0.75):
                for ws in (4, 8, 16, 32):
                    for beta in (1, 2, 4, 8):
                        got = solve(q_len, num_layers, cr, ws, beta)
                        exp = _oracle(q_len, num_layers, cr, ws, beta)
                        assert torch.equal(got, exp), (q_len, num_layers, cr, ws, beta, got, exp)


def check_monotone_decreasing(solve):
    # 合法区间的金字塔应严格递减
    out = solve(100, 5, 0.5, 4, 4).tolist()
    assert all(out[i] >= out[i + 1] for i in range(len(out) - 1)), out


def check_single_layer(solve):
    # num_layers=1：steps=0 → 只有 [round(max_num)]
    out = solve(100, 1, 0.5, 4, 2)
    exp = _oracle(100, 1, 0.5, 4, 2)
    assert torch.equal(out, exp), (out, exp)


def check_short_context_fallback(solve):
    # q_len 太小走 SnapKV 回退（q_len=4, ws=8, cr=0.5 → 区间不满足 → 每层 round(2)=2）
    out = solve(4, 3, 0.5, 8, 2)
    exp = torch.tensor([2, 2, 2])
    assert torch.equal(out, exp), out


check_example.example = True
