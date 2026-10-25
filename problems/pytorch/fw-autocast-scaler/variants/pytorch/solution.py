"""参考答案：CPU fp16 autocast + GradScaler 训练循环账——scale/unscale/step/update 次序与升降轨迹。"""

import torch

INF_STEPS = (1, 4)


def solve():
    x = torch.tensor([[1.5, -0.5]])
    w = torch.tensor([0.5, 0.25], requires_grad=True)
    y = torch.tensor(1.0)
    opt = torch.optim.SGD([w], lr=0.25)
    scaler = torch.amp.GradScaler(device="cpu", enabled=True, init_scale=16.0,
                                  growth_factor=2.0, backoff_factor=0.5, growth_interval=2)

    scale_traj, trackers, updated, losses = [], [], [], []
    out_dtype = loss_dtype = grad_dtype = ""
    double_unscale_raises = False

    for t in range(6):
        opt.zero_grad()
        with torch.amp.autocast("cpu", dtype=torch.float16):
            out = x @ w
            loss = ((out - y) ** 2).mean()
        if t == 0:
            out_dtype, loss_dtype = str(out.dtype), str(loss.dtype)
        scaler.scale(loss).backward()
        if t == 0:
            grad_dtype = str(w.grad.dtype)
            # 次序违规探针：同一 update 周期里第二次 unscale_ 必须抛
            scaler.unscale_(opt)
            try:
                scaler.unscale_(opt)
            except RuntimeError:
                double_unscale_raises = True
        if t in INF_STEPS:
            w.grad[0] = float("inf")  # 模拟溢出步
        before = w.detach().clone()
        scaler.step(opt)
        updated.append(not torch.equal(before, w.detach()))
        scaler.update()
        sd = scaler.state_dict()
        scale_traj.append(float(sd["scale"]))
        trackers.append(int(sd["_growth_tracker"]))
        losses.append(float(loss.item()))

    return {
        "out_dtype": out_dtype,
        "loss_dtype": loss_dtype,
        "grad_dtype": grad_dtype,
        "scale_traj": scale_traj,
        "trackers": trackers,
        "updated": updated,
        "losses": losses,
        "n_skipped": 6 - sum(updated),
        "final_w_flat": w.detach().reshape(-1).tolist(),
        "double_unscale_raises": double_unscale_raises,
    }
