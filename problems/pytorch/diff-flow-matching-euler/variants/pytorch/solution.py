import math
import numbers

import torch


def solve(x_init, velocity_fn, n_steps, t_start=1.0, t_end=0.0):
    if not torch.is_tensor(x_init):
        raise ValueError("x_init 必须是 torch.Tensor")
    if not callable(velocity_fn):
        raise ValueError("velocity_fn 必须是 callable")
    if isinstance(n_steps, bool) or not isinstance(n_steps, int) or n_steps < 1:
        raise ValueError("n_steps 必须是正整数")
    t0, t1 = float(t_start), float(t_end)
    if not (math.isfinite(t0) and math.isfinite(t1)):
        raise ValueError("t_start / t_end 必须是有限实数")
    if not t1 < t0:
        raise ValueError("要求 t_end < t_start（从噪声侧积回数据侧）")

    dt = (t1 - t0) / n_steps
    x = x_init.to(torch.float32)
    trajectory = [x]

    for k in range(n_steps):
        t = t0 + k * dt
        v = velocity_fn(x, t)
        if torch.is_tensor(v):
            if v.shape != x.shape:
                raise ValueError(
                    f"velocity_fn 返回形状 {tuple(v.shape)} 与输入 {tuple(x.shape)} 不一致"
                )
            step = v.to(torch.float64)
        elif isinstance(v, numbers.Real):
            step = torch.tensor(float(v), dtype=torch.float64)
        else:
            raise ValueError("velocity_fn 必须返回 torch.Tensor 或标量")
        x = (x.to(torch.float64) + dt * step).to(torch.float32)
        trajectory.append(x)

    return trajectory, trajectory[-1]
