import math

import torch

EX_FULL = {
    "running_mean_after2": [0.058750003576278687, 0.13124999403953552,
                            0.2537499964237213],
    "running_var_after2": [1.1478124856948853, 1.1478124856948853,
                           1.1478124856948853],
    "num_batches_train": 2,
    "eval_state_unchanged": True,
    "eval_out_requires_grad": True,
    "eval_out_flat": [0.41185787320137024, -1.055896282196045, 1.6299304962158203,
                      0.8785523176193237, 0.11083994060754776, -0.7035419344902039],
    "dropout_train_unique": [0.0, 1.3333333730697632],
    "n_dropout_zeros": 8,
    "dropout_eval_identity": True,
    "inference_requires_grad": False,
    "inference_backward_raises": True,
    "inference_enable_raises": True,
    "no_grad_enable_ok": True,
    "total_norm_inf": 4.0,
    "grads_after_flat": [0.7499998211860657, -0.9999997615814209, 0.0,
                         0.49999988079071045, 0.24999994039535522,
                         -0.12499997019767761],
    "clip_scaled_uniformly": True,
}

P_DROP = 0.25


def _bn_running(x1, x2, momentum=0.1):
    """纯 Python BatchNorm1d running 账：两批各按通道做 (1-m)*old + m*batch，
    batch 方差取无偏 (除以 N-1)。"""
    rm, rv = [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]
    for xb in (x1, x2):
        for c in range(3):
            col = [row[c] for row in xb]
            mean = sum(col) / len(col)
            var = sum((v - mean) ** 2 for v in col) / (len(col) - 1)
            rm[c] = (1.0 - momentum) * rm[c] + momentum * mean
            rv[c] = (1.0 - momentum) * rv[c] + momentum * var
    return rm, rv


def _oracle():
    x1 = [[v / 4.0 for v in row] for row in ([0, 1, 2], [3, 4, 5])]  # arange(6)/4 reshape(2,3)
    x2 = [[-1.0, 2.0, 0.5], [1.5, -0.5, 3.0]]
    rm, rv = _bn_running(x1, x2)
    xe = [[0.5, -1.0, 2.0], [1.0, 0.25, -0.5]]
    out_flat = []
    for row in xe:
        for c, v in enumerate(row):
            out_flat.append((v - rm[c]) / math.sqrt(rv[c] + 1e-5))
    pre = [3.0, -4.0, 0.0, 2.0, 1.0, -0.5]
    total = max(abs(g) for g in pre)
    coef = 1.0 / (total + 1e-6)
    grads_after = [g * coef for g in pre]
    return rm, rv, out_flat, total, coef, grads_after


RM, RV, EVAL_FLAT, TOTAL, COEF, GRADS_AFTER = _oracle()


def _close(got, exp, name):
    assert torch.allclose(torch.tensor(got), torch.tensor(exp),
                          atol=1e-5, rtol=1e-5), f"{name} 不符: {got} vs {exp}"


def check_example(solve):
    out = solve()
    _close(out["running_mean_after2"], EX_FULL["running_mean_after2"], "running_mean")
    _close(out["running_var_after2"], EX_FULL["running_var_after2"], "running_var")
    _close(out["eval_out_flat"], EX_FULL["eval_out_flat"], "eval_out")
    _close(out["grads_after_flat"], EX_FULL["grads_after_flat"], "grads_after")
    assert out["num_batches_train"] == 2
    assert out["n_dropout_zeros"] == 8
    assert out["dropout_train_unique"] == EX_FULL["dropout_train_unique"]
    assert out["total_norm_inf"] == 4.0
    for k in ("eval_state_unchanged", "eval_out_requires_grad",
              "dropout_eval_identity", "inference_backward_raises",
              "inference_enable_raises", "no_grad_enable_ok",
              "clip_scaled_uniformly"):
        assert out[k] is True, f"{k} 应为 True"
    assert out["inference_requires_grad"] is False


check_example.example = True


def check_matches_oracle(solve):
    out = solve()
    _close(out["running_mean_after2"], RM, "running_mean(oracle)")
    _close(out["running_var_after2"], RV, "running_var(oracle)")
    _close(out["eval_out_flat"], EVAL_FLAT, "eval_out(oracle)")
    _close(out["grads_after_flat"], GRADS_AFTER, "grads_after(oracle)")
    assert out["total_norm_inf"] == TOTAL == 4.0


def check_mode_invariants(solve):
    out = solve()
    scale = float(1.0 / (1.0 - P_DROP))
    assert len(out["dropout_train_unique"]) == 2, "ones 输入下值域必须恰为 {0, 1/(1-p)}"
    assert out["dropout_train_unique"][0] == 0.0
    assert math.isclose(out["dropout_train_unique"][1], scale, rel_tol=1e-6)
    assert 0 < out["n_dropout_zeros"] < 20, "0<p<1 时两类值都必须出现"
    assert out["dropout_eval_identity"] is True
    assert out["num_batches_train"] == 2
    assert out["eval_state_unchanged"] is True, "eval 前向不许动 running/num_batches"


def check_inference_vs_eval(solve):
    out = solve()
    assert out["eval_out_requires_grad"] is True, "eval() 不关 autograd"
    assert out["inference_requires_grad"] is False
    assert out["inference_backward_raises"] is True
    assert out["inference_enable_raises"] is True
    assert out["no_grad_enable_ok"] is True


def check_clip_uniform(solve):
    out = solve()
    assert out["clip_scaled_uniformly"] is True
    assert out["total_norm_inf"] == 4.0
    assert all(abs(g) < 1.0 for g in out["grads_after_flat"]), \
        "1e-6 兜底使裁剪后最大范数严格小于 max_norm"
    _close(out["grads_after_flat"][4:], [1.0 * COEF, -0.5 * COEF], "小梯度也被统一缩放")
    assert abs(out["grads_after_flat"][4] - 1.0) > 0.3, "不是只裁大梯度"
