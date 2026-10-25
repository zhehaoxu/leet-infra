"""参考答案：RLHF 多模型 colocate 的显存与时间线账（纯表算，不做真实前向）。"""

import torch


def solve(models, model_device, phase_model, phase_device, phase_train,
          resident, phase_time, hbm_cap, num_devices):
    M = int(models.shape[0])
    P = int(phase_model.numel())
    D = int(num_devices)
    param = models[:, 0].to(torch.long)
    opt = models[:, 1].to(torch.long)
    act = models[:, 2].to(torch.long)
    full = param + opt                       # offload/onload 整体搬运的状态
    md = model_device.to(torch.long)
    res = resident.to(torch.long)
    ini = (res[0] if P > 0 else torch.zeros(M, dtype=torch.long))

    # 1) 每阶段每设备的常驻显存
    occ = torch.zeros(P, D, dtype=torch.long)
    if P > 0:
        for m in range(M):
            on = res[:, m] == 1
            if bool(on.any()):
                occ[on, int(md[m])] += int(full[m])
        for p in range(P):
            if int(phase_train[p]) == 1:
                occ[p, int(phase_device[p])] += int(act[int(phase_model[p])])

    peak = occ.max(dim=0).values if P > 0 else torch.zeros(D, dtype=torch.long)

    # 2) offload/onload 传输量：稳态→阶段0、阶段间、末阶段→稳态（收尾回库）
    moves = torch.zeros(M, dtype=torch.long)
    if M > 0:
        flips = (res[1:] != res[:-1]).to(torch.long).sum(dim=0) if P > 1 else torch.zeros(M, dtype=torch.long)
        tail = (ini != res[-1]).to(torch.long) if P > 0 else torch.zeros(M, dtype=torch.long)
        moves = flips + tail
    offload = torch.zeros(D, dtype=torch.long)
    for m in range(M):
        offload[int(md[m])] += moves[m] * int(full[m])

    # 3) 空闲窗口：谁没被指派这阶段的时间
    busy = torch.zeros(P, D, dtype=torch.long)
    if P > 0:
        busy[torch.arange(P), phase_device.to(torch.long)] = 1
    idle = 1 - busy
    pt = phase_time.to(torch.long).reshape(-1, 1)
    idle_time = (idle * pt).sum(dim=0) if P > 0 else torch.zeros(D, dtype=torch.long)
    max_idle = torch.zeros(D, dtype=torch.long)
    times = phase_time.to(torch.long).tolist()
    for d in range(D):
        run = 0
        best = 0
        for p in range(P):
            if int(idle[p, d]) == 1:
                run += int(times[p])
                best = max(best, run)
            else:
                run = 0
        max_idle[d] = best

    feasible = peak <= int(hbm_cap)
    return peak, offload, idle_time, max_idle, feasible
