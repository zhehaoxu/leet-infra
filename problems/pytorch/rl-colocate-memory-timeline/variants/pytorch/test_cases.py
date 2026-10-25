import torch

MODELS = torch.tensor([[400, 2000, 300],
                       [400, 2000, 250],
                       [400, 0, 120],
                       [400, 0, 110]], dtype=torch.long)
MODEL_DEVICE = torch.tensor([0, 1, 0, 1], dtype=torch.long)
PHASE_MODEL = torch.tensor([0, 3, 1, 0, 2, 0], dtype=torch.long)
PHASE_DEVICE = torch.tensor([0, 1, 1, 0, 0, 0], dtype=torch.long)
PHASE_TRAIN = torch.tensor([0, 0, 0, 0, 0, 1], dtype=torch.long)
RESIDENT = torch.tensor([[1, 0, 0, 1],
                         [1, 0, 0, 1],
                         [1, 1, 0, 1],
                         [1, 1, 0, 1],
                         [1, 1, 1, 0],
                         [1, 1, 1, 0]], dtype=torch.long)
PHASE_TIME = torch.tensor([5, 2, 2, 2, 2, 4], dtype=torch.long)


def _oracle(models, model_device, phase_model, phase_device, phase_train,
            resident, phase_time, hbm_cap, num_devices):
    """独立重算：四元组 (阶段,设备,模型) 显式三重循环 + Python int；
    翻转用「稳态→各阶段→稳态」展开成一条状态序列后数相邻不等；
    最长空闲窗口先切成连续段再对每段求和。与参考解结构不同。"""
    mod = [[int(x) for x in row] for row in models.tolist()]
    mdev = [int(x) for x in model_device.tolist()]
    pmod = [int(x) for x in phase_model.tolist()]
    pdev = [int(x) for x in phase_device.tolist()]
    ptr = [int(x) for x in phase_train.tolist()]
    res = [[int(v) for v in row] for row in resident.tolist()]
    ptime = [int(x) for x in phase_time.tolist()]
    M, P, D = len(mod), len(ptime), int(num_devices)
    full = [mod[m][0] + mod[m][1] for m in range(M)]

    peak = [0] * D
    for p in range(P):
        load = [0] * D
        for m in range(M):
            if res[p][m] == 1:
                load[mdev[m]] += full[m]
        if ptr[p] == 1:
            load[pdev[p]] += mod[pmod[p]][2]
        for d in range(D):
            if load[d] > peak[d]:
                peak[d] = load[d]

    offload = [0] * D
    for m in range(M):
        ini = res[0][m] if P else 0
        seq = [ini] + [res[p][m] for p in range(P)] + [ini]
        moves = sum(1 for i in range(len(seq) - 1) if seq[i] != seq[i + 1])
        offload[mdev[m]] += moves * full[m]

    idle_time = [0] * D
    max_idle = [0] * D
    for d in range(D):
        runs = [[]]
        for p in range(P):
            if pdev[p] != d:
                runs[-1].append(ptime[p])
            else:
                runs.append([])
        idle_time[d] = sum(sum(r) for r in runs)
        max_idle[d] = max([sum(r) for r in runs] or [0])
    feasible = [peak[d] <= int(hbm_cap) for d in range(D)]

    t = lambda v: torch.tensor(v, dtype=torch.long)  # noqa: E731
    return (t(peak), t(offload), t(idle_time), t(max_idle),
            torch.tensor(feasible, dtype=torch.bool))


def _call(solve, *args):
    """薄封装：与参考解同签名，方便阅读下面的用例。"""
    return solve(*args)
# ---- 样例 ------------------------------------------------------------------------
def check_example(solve):
    peak, offload, idle, mx, feas = solve(MODELS, MODEL_DEVICE, PHASE_MODEL, PHASE_DEVICE,
                                          PHASE_TRAIN, RESIDENT, PHASE_TIME, 4000, 2)
    assert torch.equal(peak, torch.tensor([3100, 2800])), peak
    assert torch.equal(offload, torch.tensor([800, 5600])), offload
    assert torch.equal(idle, torch.tensor([4, 13])), idle
    assert torch.equal(mx, torch.tensor([4, 8])), mx
    assert torch.equal(feas, torch.tensor([True, True])), feas
    assert feas.dtype == torch.bool


check_example.example = True


def check_example_tight_cap(solve):
    """同一张表把 HBM 压到 3000：GPU0 峰值 3100 溢出、GPU1 仍可行。"""
    peak, _o, _i, _m, feas = solve(MODELS, MODEL_DEVICE, PHASE_MODEL, PHASE_DEVICE,
                                   PHASE_TRAIN, RESIDENT, PHASE_TIME, 3000, 2)
    assert torch.equal(feas, torch.tensor([False, True])), feas
    assert int(peak[0]) == 3100


check_example_tight_cap.example = True


# ---- 退化探针 --------------------------------------------------------------------
def check_all_resident_no_transfer(solve):
    """resident 全 1 → 没有任何翻转 → offload 全 0；稳态=末态也自洽。"""
    res = torch.ones(RESIDENT.shape, dtype=torch.long)
    _p, offload, _i, _m, _f = solve(MODELS, MODEL_DEVICE, PHASE_MODEL, PHASE_DEVICE,
                                    PHASE_TRAIN, res, PHASE_TIME, 10 ** 9, 2)
    assert torch.equal(offload, torch.zeros(2, dtype=torch.long)), offload


def check_all_offloaded_only_activation(solve):
    """resident 全 0 → 常驻占用 0、翻转 0；只有 train 阶段的 act 落在 phase_device 上。"""
    res = torch.zeros(RESIDENT.shape, dtype=torch.long)
    peak, offload, _i, _m, feas = solve(MODELS, MODEL_DEVICE, PHASE_MODEL, PHASE_DEVICE,
                                        PHASE_TRAIN, res, PHASE_TIME, 300, 2)
    assert torch.equal(peak, torch.tensor([300, 0])), peak    # p5 在 GPU0 上摊入 actor 的 act
    assert torch.equal(offload, torch.zeros(2, dtype=torch.long)), offload
    assert torch.equal(feas, torch.tensor([True, True])), feas


def check_single_device_no_idle(solve):
    """D=1 且全常驻：所有阶段都在这台卡上 → 空闲账与搬运账全 0；峰值 = Σ(param+opt) + 训练 act。"""
    peak, offload, idle, mx, _f = solve(MODELS, torch.zeros(4, dtype=torch.long),
                                        PHASE_MODEL, torch.zeros(6, dtype=torch.long),
                                        PHASE_TRAIN, torch.ones_like(RESIDENT), PHASE_TIME, 10 ** 9, 1)
    assert torch.equal(idle, torch.zeros(1, dtype=torch.long)), idle
    assert torch.equal(mx, torch.zeros(1, dtype=torch.long)), mx
    assert torch.equal(offload, torch.zeros(1, dtype=torch.long)), offload
    assert int(peak[0]) == 400 * 4 + 2000 * 2 + 300, peak       # 5600 常驻 + 300 激活


def check_one_device_gated_by_resident(solve):
    """D=1 但按表 offload：峰值只算当阶段常驻的模型（p5 = actor+critic+ref + act）。"""
    peak, offload, _i, _m, _f = solve(MODELS, torch.zeros(4, dtype=torch.long),
                                      PHASE_MODEL, torch.zeros(6, dtype=torch.long),
                                      PHASE_TRAIN, RESIDENT, PHASE_TIME, 10 ** 9, 1)
    assert int(peak[0]) == 2400 + 2400 + 400 + 300, peak
    assert torch.equal(offload, torch.tensor([6400])), offload  # critic 4800 + ref 800 + reward 800


def check_activation_lands_on_phase_device(solve):
    """被更新的模型属于 GPU0，但阶段跑在 GPU1 且该模型此刻 offload → act 只记在 GPU1。"""
    models = torch.tensor([[1000, 500, 77]], dtype=torch.long)
    peak, offload, idle, mx, _f = solve(
        models, torch.tensor([0]), torch.tensor([0, 0]),
        torch.tensor([1, 0]), torch.tensor([1, 0]),
        torch.tensor([[0], [0]], dtype=torch.long),
        torch.tensor([3, 4]), 10 ** 9, 2)
    assert torch.equal(peak, torch.tensor([0, 77])), peak       # 激活不在模型所属卡上
    assert torch.equal(offload, torch.zeros(2, dtype=torch.long)), offload
    assert torch.equal(idle, torch.tensor([3, 4])), idle        # 各空闲一阶段
    assert torch.equal(mx, torch.tensor([3, 4])), mx


def check_offload_moves_are_even(solve):
    """时间线首尾回到同一稳态 ⇒ 每个模型的翻转次数为偶数 ⇒ 每台设备的搬运字节和为偶数。"""
    _p, offload, _i, _m, _f = solve(MODELS, MODEL_DEVICE, PHASE_MODEL, PHASE_DEVICE,
                                    PHASE_TRAIN, RESIDENT, PHASE_TIME, 10 ** 9, 2)
    assert all(int(v) % 2 == 0 for v in offload.tolist()), offload


def check_zero_phases(solve):
    """P=0：没有阶段 → 整数账全 0、feasible 全 True（0 占用一定不超 cap）。"""
    peak, offload, idle, mx, feas = solve(MODELS, MODEL_DEVICE,
                                          torch.zeros(0, dtype=torch.long),
                                          torch.zeros(0, dtype=torch.long),
                                          torch.zeros(0, dtype=torch.long),
                                          torch.zeros(0, 4, dtype=torch.long),
                                          torch.zeros(0, dtype=torch.long), 10, 3)
    for v in (peak, offload, idle, mx):
        assert v.numel() == 3 and int(v.abs().sum()) == 0, v
    assert torch.equal(feas, torch.ones(3, dtype=torch.bool)), feas


def check_single_phase_steady_no_restore_move(solve):
    """P=1：稳态就是这一阶段本身 → 收尾翻转必为 0，offload 恒为 0。"""
    res = torch.tensor([[1, 0, 1, 0]], dtype=torch.long)
    _p, offload, _i, _m, _f = solve(MODELS, MODEL_DEVICE, torch.tensor([2]),
                                    torch.tensor([0]), torch.tensor([1]), res,
                                    torch.tensor([7]), 10 ** 9, 2)
    assert torch.equal(offload, torch.zeros(2, dtype=torch.long)), offload


def check_infeasible_at_zero_cap(solve):
    _p, _o, _i, _m, feas = solve(MODELS, MODEL_DEVICE, PHASE_MODEL, PHASE_DEVICE,
                                 PHASE_TRAIN, RESIDENT, PHASE_TIME, 0, 2)
    assert torch.equal(feas, torch.tensor([False, False])), feas


# ---- 与独立 oracle 对拍 ----------------------------------------------------------
def check_matches_oracle_battery(solve):
    torch.manual_seed(17)
    battery = [
        (MODELS, MODEL_DEVICE, PHASE_MODEL, PHASE_DEVICE, PHASE_TRAIN, RESIDENT, PHASE_TIME, 4000, 2),
        (MODELS, MODEL_DEVICE, PHASE_MODEL, PHASE_DEVICE, PHASE_TRAIN,
         torch.ones_like(RESIDENT), PHASE_TIME, 5000, 2),
        (MODELS, torch.tensor([0, 0, 0, 0]), PHASE_MODEL, torch.zeros(6, dtype=torch.long),
         PHASE_TRAIN, RESIDENT, PHASE_TIME, 12000, 1),
        (torch.tensor([[10, 0, 5], [20, 30, 7]], dtype=torch.long),
         torch.tensor([2, 1]), torch.tensor([0, 1, 0, 1, 0], dtype=torch.long),
         torch.tensor([2, 1, 0, 0, 1], dtype=torch.long),
         torch.tensor([0, 1, 1, 0, 0], dtype=torch.long),
         torch.tensor([[1, 1], [1, 0], [0, 0], [0, 1], [1, 1]], dtype=torch.long),
         torch.tensor([1, 2, 3, 4, 5], dtype=torch.long), 40, 3),
        (torch.tensor([[7, 7, 7]], dtype=torch.long), torch.tensor([0]),
         torch.tensor([0, 0]), torch.tensor([0, 0]), torch.tensor([1, 0], dtype=torch.long),
         torch.tensor([[1], [0]], dtype=torch.long), torch.tensor([2, 3]), 21, 1),
    ]
    for args in battery:
        got = solve(*args)
        ref = _oracle(*args)
        for i in range(5):
            if i < 4:
                assert torch.equal(got[i], ref[i]), (args[0][0].tolist(), i, got[i], ref[i])
            else:
                assert torch.equal(got[i], ref[i]), (i, got[i], ref[i])

    for trial in range(12):
        M = 1 + trial % 5
        P = 1 + trial % 7
        D = 1 + trial % 3
        models = torch.randint(1, 50, (M, 3), dtype=torch.long) * 10
        models[:, 1] = (trial % 2) * models[:, 1]
        mdev = torch.randint(0, D, (M,), dtype=torch.long)
        pmod = torch.randint(0, M, (P,), dtype=torch.long)
        pdev = torch.randint(0, D, (P,), dtype=torch.long)
        ptr = torch.randint(0, 2, (P,), dtype=torch.long)
        res = torch.randint(0, 2, (P, M), dtype=torch.long)
        ptime = torch.randint(1, 9, (P,), dtype=torch.long)
        cap = int(models.sum()) // 3
        args = (models, mdev, pmod, pdev, ptr, res, ptime, cap, D)
        got = solve(*args)
        ref = _oracle(*args)
        for i in range(5):
            assert torch.equal(got[i], ref[i]), (trial, i, got[i], ref[i])


def check_timeline_invariants(solve):
    """时间线自洽：空闲+忙碌=总时长；max_idle ≤ idle_time；peak ≥ 任一常驻模型的完整态。"""
    total = int(PHASE_TIME.sum())
    peak, offload, idle, mx, feas = solve(MODELS, MODEL_DEVICE, PHASE_MODEL, PHASE_DEVICE,
                                          PHASE_TRAIN, RESIDENT, PHASE_TIME, 3100, 2)
    for d in range(2):
        busy = int(sum(int(PHASE_TIME[p]) for p in range(6) if int(PHASE_DEVICE[p]) == d))
        assert busy + int(idle[d]) == total, (d, busy, idle[d])
        assert int(mx[d]) <= int(idle[d]), (d, mx[d], idle[d])
    assert int(peak[0]) >= int(MODELS[0, 0] + MODELS[0, 1]), peak
    assert torch.equal(feas, torch.tensor([True, True])), feas
    assert offload.dtype == torch.long and all(int(v) % 2 == 0 for v in offload.tolist())
