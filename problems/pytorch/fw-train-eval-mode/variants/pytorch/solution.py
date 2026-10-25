"""参考答案：train/eval 模式、inference_mode 与 clip_grad_norm 的行为账。"""

import torch
import torch.nn as nn

P_DROP = 0.25


def solve():
    # —— 阶段 A：BatchNorm1d 两次 train 前向，running 统计账
    bn = nn.BatchNorm1d(3)
    bn.train()
    x1 = torch.arange(6).reshape(2, 3) / 4.0
    x2 = torch.tensor([[-1.0, 2.0, 0.5], [1.5, -0.5, 3.0]])
    bn(x1)
    bn(x2)
    num_batches_train = int(bn.num_batches_tracked.item())

    # —— 阶段 B：eval 前向，running 统计与计数一律不动
    state_before = {k: v.clone() for k, v in bn.state_dict().items()}
    bn.eval()
    xe = torch.tensor([[0.5, -1.0, 2.0], [1.0, 0.25, -0.5]])
    out_eval = bn(xe)
    eval_state_unchanged = all(
        torch.equal(state_before[k], v) for k, v in bn.state_dict().items())
    eval_out_requires_grad = bool(out_eval.requires_grad)

    # —— 阶段 C：Dropout 开关账（固定 seed）
    torch.manual_seed(7)
    drop = nn.Dropout(P_DROP)
    xd = torch.ones(4, 5)
    drop.train()
    yd = drop(xd)
    vals = yd.reshape(-1)
    scale = float(1.0 / (1.0 - P_DROP))
    n_zeros = int((vals == 0).sum().item())
    dropout_train_unique = sorted({float(v) for v in vals.tolist()})
    drop.eval()
    dropout_eval_identity = bool(torch.equal(drop(xd), xd))

    # —— 阶段 D：eval() ≠ no_grad；inference_mode 与 no_grad 的张量性质
    xg = torch.tensor([1.0, 2.0], requires_grad=True)
    with torch.inference_mode():
        yi = xg * 2.0
    inference_requires_grad = bool(yi.requires_grad)
    try:
        yi.sum().backward()
        inference_backward_raises = False
    except RuntimeError:
        inference_backward_raises = True
    try:
        yi.requires_grad_(True)
        inference_enable_raises = False
    except RuntimeError:
        inference_enable_raises = True
    with torch.no_grad():
        zn = xg * 2.0
    try:
        zn.requires_grad_(True)
        no_grad_enable_ok = bool(zn.requires_grad)
    except RuntimeError:
        no_grad_enable_ok = False

    # —— 阶段 E：clip_grad_norm_(norm_type=inf) 均匀缩放账
    w1 = nn.Parameter(torch.tensor([[3.0, -4.0], [0.0, 2.0]]))
    w2 = nn.Parameter(torch.tensor([1.0, -0.5]))
    w1.grad = w1.detach().clone()
    w2.grad = w2.detach().clone()
    total = torch.nn.utils.clip_grad_norm_([w1, w2], max_norm=1.0,
                                           norm_type=float("inf"))
    total_norm_inf = float(total.item())
    coef = 1.0 / (total_norm_inf + 1e-6)
    pre_big = torch.tensor([[3.0, -4.0], [0.0, 2.0]])
    pre_small = torch.tensor([1.0, -0.5])
    clip_scaled_uniformly = bool(
        torch.allclose(w1.grad, pre_big * coef, atol=1e-5, rtol=1e-5)
        and torch.allclose(w2.grad, pre_small * coef, atol=1e-5, rtol=1e-5))
    grads_after_flat = w1.grad.reshape(-1).tolist() + w2.grad.tolist()

    return {
        "running_mean_after2": bn.running_mean.tolist(),
        "running_var_after2": bn.running_var.tolist(),
        "num_batches_train": num_batches_train,
        "eval_state_unchanged": eval_state_unchanged,
        "eval_out_requires_grad": eval_out_requires_grad,
        "eval_out_flat": out_eval.detach().reshape(-1).tolist(),
        "dropout_train_unique": dropout_train_unique,
        "n_dropout_zeros": n_zeros,
        "dropout_eval_identity": dropout_eval_identity,
        "inference_requires_grad": inference_requires_grad,
        "inference_backward_raises": inference_backward_raises,
        "inference_enable_raises": inference_enable_raises,
        "no_grad_enable_ok": no_grad_enable_ok,
        "total_norm_inf": total_norm_inf,
        "grads_after_flat": grads_after_flat,
        "clip_scaled_uniformly": clip_scaled_uniformly,
    }
