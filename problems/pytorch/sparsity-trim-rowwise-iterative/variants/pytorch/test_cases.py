import numpy as np
import torch

KEYS = {"sparsities", "keep_mask", "pruned_weight", "baseline_metric", "best_metric",
        "learning_rate", "best_iter", "starting_var", "ending_var"}
_POS = (0.01, 0.02, 0.04, 0.08, 0.12, 0.16)
_NEG = (-0.01, -0.02, -0.04)


def _oracle(W, X, act, target, iterations=10, pos=_POS, neg=_NEG, forbid_neg=False):
    """numpy 独立重算：稳定排序 → 余弦反馈迭代 → lr 网格择优 → 后处理。"""
    W = np.asarray(W, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    act = np.asarray(act, dtype=np.float64)
    out_features, in_features = W.shape
    metric = np.abs(W) * np.sqrt(act)[None, :]
    order = np.argsort(metric, axis=1, kind="stable")
    unpruned = W @ X

    def prune(sparsities):
        sparsities = np.asarray(sparsities, dtype=np.float64)
        num_prune = (in_features * sparsities).astype(np.int64)
        keep = np.ones_like(W, dtype=bool)
        for i in range(out_features):
            count = max(0, min(in_features, int(num_prune[i])))
            keep[i, order[i, :count]] = False
        return W * keep, keep

    def flat_cos(a, b):
        a = a.ravel() / (np.linalg.norm(a) + 1e-8)
        b = b.ravel() / (np.linalg.norm(b) + 1e-8)
        return float(np.clip(a @ b, -1.0, 1.0))

    def row_cos(a, b):
        a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
        b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
        return np.clip((a * b).sum(axis=1), -1.0, 1.0)

    def run(learning_rate):
        x = np.full(out_features, target)
        pruned = prune(x)[0] @ X
        best_metric = flat_cos(unpruned, pruned)
        best_x = x.copy()
        best_iter = 0
        for step in range(iterations):
            qm = row_cos(unpruned, pruned)
            delta = (qm - qm.min()) / (qm.max() - qm.min() + 1e-6)
            delta = delta * (learning_rate * 2.0)
            delta = delta - delta.mean()
            x = x + delta
            x = np.minimum(x, 0.95)
            x = np.maximum(x, 0.0)
            x = x - x.mean() + target
            pruned = prune(x)[0] @ X
            score = flat_cos(unpruned, pruned)
            if score > best_metric:
                best_metric = score
                best_x = x.copy()
                best_iter = step
        return best_x, best_metric, best_iter

    pruned0 = prune(np.full(out_features, target))[0] @ X
    baseline = flat_cos(unpruned, pruned0)
    start_var = float(np.var(row_cos(unpruned, pruned0), ddof=1))
    best_metric = baseline
    best_sparsities = np.full(out_features, target)
    best_lr = None
    best_iter = 0
    best_this = -np.inf
    for lr in pos:
        sparsities, score, step = run(float(lr))
        if score > best_metric:
            best_metric, best_sparsities, best_lr, best_iter = score, sparsities, float(lr), step
        if score > best_this:
            best_this = score
        else:
            break
    if best_lr is None and not forbid_neg:
        best_this = -np.inf
        for lr in neg:
            sparsities, score, step = run(float(lr))
            if score > best_metric:
                best_metric, best_sparsities, best_lr, best_iter = score, sparsities, float(lr), step
            if score > best_this:
                best_this = score
            else:
                break
    end_var = float(np.var(row_cos(unpruned, prune(best_sparsities)[0] @ X), ddof=1))

    final = np.clip(best_sparsities, 0.0, 0.95)
    final = final - final.mean() + target
    pruned_weight, keep = prune(final)
    return {
        "sparsities": final,
        "keep_mask": keep,
        "pruned_weight": pruned_weight,
        "baseline_metric": baseline,
        "best_metric": best_metric,
        "learning_rate": best_lr,
        "best_iter": best_iter,
        "starting_var": start_var,
        "ending_var": end_var,
    }


def _check(solve, dtype, W, X, act, target, iterations=10, **kwargs):
    wt = torch.tensor(W, dtype=dtype)
    xt = torch.tensor(X, dtype=dtype)
    at = torch.tensor(act, dtype=dtype)
    snaps = (wt.clone(), xt.clone(), at.clone())
    got = solve(wt, xt, at, target, iterations, **kwargs)
    want = _oracle(np.asarray(wt, dtype=np.float64), np.asarray(xt, dtype=np.float64),
                   np.asarray(at, dtype=np.float64), target, iterations,
                   pos=kwargs.get("lr_grid", _POS), neg=kwargs.get("neg_lr_grid", _NEG),
                   forbid_neg=kwargs.get("forbid_neg_lr", False))
    assert isinstance(got, dict) and set(got) == KEYS, f"键不对：{sorted(got)}"
    for tensor, snap in zip((wt, xt, at), snaps):
        assert torch.equal(tensor, snap), "输入被就地修改"
    tol = 1e-9
    assert got["sparsities"].dtype == torch.float64
    assert np.allclose(got["sparsities"].numpy(), want["sparsities"], atol=tol, rtol=0)
    assert got["keep_mask"].dtype == torch.bool
    assert got["keep_mask"].numpy().tolist() == want["keep_mask"].tolist()
    assert got["pruned_weight"].dtype == dtype
    assert np.allclose(got["pruned_weight"].to(torch.float64).numpy(),
                       want["pruned_weight"], atol=tol, rtol=0)
    for key in ("baseline_metric", "best_metric", "starting_var", "ending_var"):
        assert isinstance(got[key], float), f"{key} 必须是 python float"
        assert abs(got[key] - want[key]) < tol, f"{key}: {got[key]} != {want[key]}"
    assert got["learning_rate"] == want["learning_rate"], "learning_rate 不对"
    assert got["best_iter"] == want["best_iter"], "best_iter 不对"
    return got


W = [[3.0, 2.0, 1.0, 0.0], [0.0, 1.0, 2.0, 3.0],
     [1.0, 1.0, 1.0, 1.0], [4.0, 0.0, 0.0, 1.0]]
X = [[1.0, 0.0, 1.0], [0.5, 1.0, -1.0], [2.0, 0.5, 1.0], [1.0, 1.5, -1.0]]
ACT = [2.0 / 3.0, 0.75, 1.75, 17.0 / 12.0]


def check_example(solve):
    got = _check(solve, torch.float64, W, X, ACT, 0.5, iterations=5)
    assert np.allclose(got["sparsities"].numpy(),
                       [0.48964132, 0.49108361, 0.50963753, 0.50963754], atol=1e-8)
    assert got["keep_mask"].tolist() == [[True, True, True, False],
                                         [False, True, True, True],
                                         [False, False, True, True],
                                         [True, False, False, True]]
    assert got["learning_rate"] == 0.01 and got["best_iter"] == 0
    assert abs(got["baseline_metric"] - 0.985699645692995) < 1e-10
    assert abs(got["best_metric"] - 0.9928880901816941) < 1e-10


check_example.example = True


def check_iterations_zero(solve):
    got = _check(solve, torch.float64, W, X, ACT, 0.5, iterations=0)
    assert np.allclose(got["sparsities"].numpy(), [0.5] * 4, atol=1e-12)
    assert got["best_metric"] == got["baseline_metric"]
    assert got["learning_rate"] is None and got["best_iter"] == 0
    assert got["starting_var"] == got["ending_var"]


def check_matches_oracle(solve):
    rng = np.random.default_rng(13)
    for shape in [(3, 6, 5), (4, 8, 7), (5, 5, 9)]:
        out_features, in_features, tokens = shape
        w = rng.normal(0, 2, (out_features, in_features))
        x = rng.normal(0, 1, (in_features, tokens))
        act = (x ** 2).mean(axis=1)
        for target in [0.4, 0.5, 0.7]:
            _check(solve, torch.float64, w.tolist(), x.tolist(), act.tolist(),
                   target, iterations=6)


def check_mask_count_rule(solve):
    got = _check(solve, torch.float64, W, X, ACT, 0.6, iterations=4)
    sparsities = got["sparsities"].numpy()
    for row, sparsity in enumerate(sparsities):
        expected = max(0, min(4, int(4 * sparsity)))
        assert int((~got["keep_mask"][row]).sum()) == expected


def check_monotone_metric(solve):
    got = _check(solve, torch.float64, W, X, ACT, 0.5, iterations=5)
    assert got["best_metric"] >= got["baseline_metric"] - 1e-12
    if got["learning_rate"] is not None:
        assert got["best_metric"] > got["baseline_metric"]
    assert isinstance(got["ending_var"], float)


def check_forbid_neg_lr(solve):
    got = _check(solve, torch.float64, W, X, ACT, 0.5, iterations=5, forbid_neg_lr=True)
    assert got["learning_rate"] in (None,) + _POS


def check_dtype_and_errors(solve):
    got = _check(solve, torch.float32, W, X, ACT, 0.5, iterations=3)
    assert got["pruned_weight"].dtype == torch.float32
    assert got["sparsities"].dtype == torch.float64

    base = [torch.ones(3, 4), torch.ones(4, 2), torch.ones(4)]
    cases = [
        (torch.ones(4), base[1], base[2], 0.5),                  # weight 非二维
        (torch.ones(1, 4), base[1], base[2], 0.5),               # out < 2
        (base[0], torch.ones(4), base[2], 0.5),                  # inputs 非二维
        (base[0], torch.ones(3, 2), base[2], 0.5),               # inputs 行数不对
        (base[0], base[1], torch.ones(3), 0.5),                  # 激活长度不对
        (base[0], base[1], base[2].double(), 0.5),               # dtype 不一致
        (base[0].int(), base[1].int(), base[2].int(), 0.5),      # 非浮点
        (base[0], base[1], base[2], 0.0),                        # target 越界
        (base[0], base[1], base[2], 1.0),
        (base[0], base[1], base[2], True),                       # bool
    ]
    for index, (weight, inputs, act, target) in enumerate(cases):
        try:
            solve(weight, inputs, act, target)
        except ValueError:
            continue
        raise AssertionError(f"case {index}: 期望 ValueError")

    for kwargs in [{"iterations": -1}, {"qmetric": "mse"}, {"qmetric_dimwise": "mse"},
                   {"lr_grid": 0.01}, {"neg_lr_grid": None}]:
        try:
            solve(*base, 0.5, **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"kwargs {kwargs}: 期望 ValueError")
