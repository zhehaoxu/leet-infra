import numpy as np
import torch


def _oracle(wq, wk, wv, wo, gwq, gwk, gwv, gwo, head_dim,
            mode="param_first", group_reduction="sum", num_prune=1):
    """独立 numpy 重算 LLM-Pruner 组粒度泰勒头重要性。"""
    W = {"q": wq.detach().cpu().numpy().astype(np.float64),
         "k": wk.detach().cpu().numpy().astype(np.float64),
         "v": wv.detach().cpu().numpy().astype(np.float64),
         "o": wo.detach().cpu().numpy().astype(np.float64)}
    G = {"q": gwq.detach().cpu().numpy().astype(np.float64),
         "k": gwk.detach().cpu().numpy().astype(np.float64),
         "v": gwv.detach().cpu().numpy().astype(np.float64),
         "o": gwo.detach().cpu().numpy().astype(np.float64)}

    d_out, d_in = W["q"].shape
    n_heads = d_in // head_dim
    sal = {n: W[n] * G[n] for n in W}

    def block_sum(vec):
        return vec.reshape(n_heads, head_dim).sum(axis=1)

    if mode == "param_first":
        contribs = [block_sum(np.abs(sal["o"]).sum(axis=1)),
                    block_sum(np.abs(sal["q"]).sum(axis=0)),
                    block_sum(np.abs(sal["k"]).sum(axis=0)),
                    block_sum(np.abs(sal["v"]).sum(axis=0))]
    else:
        contribs = [block_sum(np.abs(sal["o"].sum(axis=1))),
                    block_sum(np.abs(sal["q"].sum(axis=0))),
                    block_sum(np.abs(sal["k"].sum(axis=0))),
                    block_sum(np.abs(sal["v"].sum(axis=0)))]

    stacked = np.stack(contribs, axis=0)
    imp = stacked.sum(axis=0) if group_reduction == "sum" else stacked.mean(axis=0)
    order = np.argsort(imp, kind="stable")
    pruned = sorted(int(i) for i in order[:num_prune])
    return {"head_importance": torch.tensor(imp, dtype=torch.float64),
            "per_source": torch.tensor(stacked, dtype=torch.float64),
            "pruned_heads": pruned}


def check_example_all_positive(solve):
    """手算用例：全正权重，param_first 与 vectorize 同结果。"""
    w = torch.tensor([[float(i * 4 + j + 1) for j in range(4)] for i in range(4)])
    g = torch.ones(4, 4)
    got = solve(w, w, w, w, g, g, g, g, head_dim=2)
    exp = _oracle(w, w, w, w, g, g, g, g, head_dim=2)
    # o 行 [10,26,42,58] → 头块 [36,100]；q/k/v 列 [28,32,36,40] → [60,76]
    assert torch.allclose(got["head_importance"], torch.tensor([216.0, 328.0], dtype=torch.float64))
    assert torch.allclose(got["per_source"], exp["per_source"])
    assert got["pruned_heads"] == [0]
    assert got["n_heads"] == 2


def check_vectorize_sign_cancellation(solve):
    """vectorize 先求和再 abs：块内正负抵消后重要性骤降。"""
    wq = torch.tensor([[1.0, -3.0, 2.0, 2.0],
                       [1.0, 3.0, -2.0, 2.0],
                       [1.0, -3.0, 2.0, 2.0],
                       [1.0, 3.0, -2.0, 2.0]])
    zero = torch.zeros(4, 4)
    g = torch.ones(4, 4)
    got_param = solve(wq, zero, zero, zero, g, g, g, g, head_dim=2,
                      mode="param_first")
    got_vec = solve(wq, zero, zero, zero, g, g, g, g, head_dim=2,
                    mode="vectorize")
    # param_first：|w| 列和 [4,12,8,8] → 头块 [16,16]（并列）
    assert torch.allclose(got_param["head_importance"], torch.tensor([16.0, 16.0], dtype=torch.float64))
    # vectorize：|列和| = [4,0,0,8] → 头块 [4,8]
    assert torch.allclose(got_vec["head_importance"], torch.tensor([4.0, 8.0], dtype=torch.float64))
    # 两者都剪头 0，但 vectorize 数值不同
    assert got_param["pruned_heads"] == [0] and got_vec["pruned_heads"] == [0]


def check_mode_flips_selection(solve):
    """param_first 与 vectorize 选出不同的头。"""
    # d=2, head_dim=1, n_heads=2：头 0 的 q 列 [3, -3] 绝对值和 6、求和后 abs=0（抵消）
    wq = torch.tensor([[3.0, 2.0], [-3.0, 0.0]])
    zero = torch.zeros(2, 2)
    g = torch.ones(2, 2)
    p = solve(wq, zero, zero, zero, g, g, g, g, head_dim=1, mode="param_first")
    v = solve(wq, zero, zero, zero, g, g, g, g, head_dim=1, mode="vectorize")
    # param_first 列分值 [6, 2] → imp [6, 2] → 剪头 1
    assert torch.allclose(p["head_importance"], torch.tensor([6.0, 2.0], dtype=torch.float64))
    assert p["pruned_heads"] == [1]
    # vectorize 列分值 [|3-3|, 2] = [0, 2] → imp [0, 2] → 剪头 0
    assert torch.allclose(v["head_importance"], torch.tensor([0.0, 2.0], dtype=torch.float64))
    assert v["pruned_heads"] == [0]


def check_tiebreak_and_extremes(solve):
    """并列 stable 取小索引；num_prune=0 空、=n_heads 全剪。"""
    w = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    g = torch.ones(2, 2)
    # 对称结构 → 两头 importance 相等 → stable 取头 0
    got1 = solve(w.t().contiguous(), w, w, w, g, g, g, g, head_dim=1, num_prune=1)
    got0 = solve(w, w, w, w, g, g, g, g, head_dim=1, num_prune=0)
    gotall = solve(w, w, w, w, g, g, g, g, head_dim=1, num_prune=2)
    assert got0["pruned_heads"] == []
    assert sorted(gotall["pruned_heads"]) == [0, 1]
    assert len(got1["pruned_heads"]) == 1


def check_mean_reduction_scales(solve):
    """mean 与 sum 只差常数因子，选择不变。"""
    torch.manual_seed(3)
    wq, wk, wv, wo = (torch.randn(6, 6) for _ in range(4))
    gq, gk, gv, go = (torch.randn(6, 6) for _ in range(4))
    s = solve(wq, wk, wv, wo, gq, gk, gv, go, head_dim=2,
              group_reduction="sum", num_prune=1)
    m = solve(wq, wk, wv, wo, gq, gk, gv, go, head_dim=2,
              group_reduction="mean", num_prune=1)
    assert torch.allclose(s["head_importance"], 4.0 * m["head_importance"], atol=1e-12)
    assert s["pruned_heads"] == m["pruned_heads"]


def check_random_against_oracle(solve):
    """随机形状双模式对拍。"""
    torch.manual_seed(17)
    for _ in range(6):
        d = int(torch.randint(2, 7, (1,))) * 2
        hd = int(torch.randint(1, 3, (1,)))
        if d % hd != 0:
            hd = 1
        n = d // hd
        mats = [torch.randn(d, d) for _ in range(4)]
        grads = [torch.randn(d, d) for _ in range(4)]
        mode = "param_first" if bool(torch.randint(0, 2, (1,))) else "vectorize"
        k = int(torch.randint(0, n + 1, (1,)))
        got = solve(*mats, *grads, head_dim=hd, mode=mode, num_prune=k)
        exp = _oracle(*mats, *grads, head_dim=hd, mode=mode, num_prune=k)
        assert torch.allclose(got["head_importance"], exp["head_importance"],
                              rtol=1e-9, atol=1e-12)
        assert got["pruned_heads"] == exp["pruned_heads"]
        assert got["n_heads"] == n


def check_validation_errors(solve):
    """非法输入全部抛 ValueError。"""
    w = torch.randn(4, 4)
    g = torch.randn(4, 4)
    bad = [
        (lambda: solve(w, w, w, torch.randn(4, 5), g, g, g, g, 2), "形状不一致"),
        (lambda: solve(w, w, w, w, g, g, g, torch.randn(4, 5), 2), "梯度形状不一致"),
        (lambda: solve(w, w, w, w, g, g, g, g.to(torch.float64), 2), "梯度 dtype 不一致"),
        (lambda: solve(w.to(torch.float64), w, w, w, g, g, g, g, 2), "权重 dtype 不一致"),
        (lambda: solve(w, w, w, w, g, g, g, g, 3), "head_dim 不整除"),
        (lambda: solve(w, w, w, w, g, g, g, g, 0), "head_dim 非正"),
        (lambda: solve(w, w, w, w, g, g, g, g, 2, mode="second"), "mode 非法"),
        (lambda: solve(w, w, w, w, g, g, g, g, 2, group_reduction="max"), "reduction 非法"),
        (lambda: solve(w, w, w, w, g, g, g, g, 2, num_prune=5), "num_prune 超头数"),
        (lambda: solve(w, w, w, w, g, g, g, g, 2, num_prune=-1), "num_prune 负"),
        (lambda: solve(torch.randn(4, 6), w, w, w, g, g, g, g, 2), "out≠in"),
        (lambda: solve(torch.randn(4,), w, w, w, g, g, g, g, 2), "一维权重"),
    ]
    for fn, _ in bad:
        try:
            fn()
        except ValueError:
            pass
        else:
            raise AssertionError(f"用例未抛错：{bad[[f for f, _ in bad].index(fn)][1]}")
