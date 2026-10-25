import torch

INIT_SCALE = 16.0
GROWTH_INTERVAL = 2


def _oracle():
    """独立重算：纯 Python 实现 GradScaler 状态机 + dyadic 精确数值前反向。
    所有值均为二进制可精确表示的小数（fp16 无舍入损失），故可用 float 直接对拍。"""
    scale, tracker = INIT_SCALE, 0
    w = [0.5, 0.25]
    scale_traj, trackers, updated, losses = [], [], [], []
    for t in range(6):
        out = 1.5 * w[0] - 0.5 * w[1]
        loss = (out - 1.0) ** 2
        losses.append(loss)
        g = [2.0 * (out - 1.0) * 1.5, 2.0 * (out - 1.0) * (-0.5)]
        has_inf = t in (1, 4)  # 题面注入溢出步
        if not has_inf:
            w = [w[0] - 0.25 * g[0], w[1] - 0.25 * g[1]]
        updated.append(not has_inf)
        if has_inf:
            scale *= 0.5
            tracker = 0
        else:
            tracker += 1
            if tracker == GROWTH_INTERVAL:
                scale *= 2.0
                tracker = 0
        scale_traj.append(scale)
        trackers.append(tracker)
    return {
        "scale_traj": scale_traj,
        "trackers": trackers,
        "updated": updated,
        "losses": losses,
        "final_w": w,
        "n_skipped": sum(1 for u in updated if not u),
    }


EXP = _oracle()


def check_example(solve):
    out = solve()
    assert out["scale_traj"] == EXP["scale_traj"], f"scale 轨迹不符: {out['scale_traj']}"
    assert out["trackers"] == EXP["trackers"], f"growth tracker 不符: {out['trackers']}"
    assert out["updated"] == EXP["updated"]
    assert out["n_skipped"] == 2
    assert out["out_dtype"] == "torch.float16"
    assert out["loss_dtype"] == "torch.float16"
    assert out["grad_dtype"] == "torch.float32"


check_example.example = True


def check_matches_oracle(solve):
    out = solve()
    assert torch.allclose(torch.tensor(out["losses"]), torch.tensor(EXP["losses"]),
                          atol=1e-5, rtol=1e-5), f"loss 轨迹不符: {out['losses']}"
    assert torch.allclose(torch.tensor(out["final_w_flat"]), torch.tensor(EXP["final_w"]),
                          atol=1e-5, rtol=1e-5), \
        f"权重轨迹不符: {out['final_w_flat']} vs {EXP['final_w']}"


def check_skipped_step_freezes_weights(solve):
    # inf 步（1、4）权重逐位不动：跳过时优化器没有被调用
    out = solve()
    assert out["updated"][1] is False and out["updated"][4] is False
    assert all(u is True for i, u in enumerate(out["updated"]) if i not in (1, 4))
    # 反证「未更新则 loss 不变」：t1 与 t2 同 loss，t4 与 t5 同 loss
    assert out["losses"][1] == out["losses"][2]
    assert out["losses"][4] == out["losses"][5]


def check_growth_and_backoff(solve):
    # 升降账：初始 16；t1 退避到 8；连续两个成功步（t2,t3）触发翻倍回 16；t4 再退 8
    out = solve()
    assert out["scale_traj"][0] == 16.0
    assert out["scale_traj"][1] == 8.0 and out["trackers"][1] == 0
    assert out["scale_traj"][3] == 16.0 and out["trackers"][3] == 0
    assert out["scale_traj"][4] == 8.0


def check_unscale_order_guard(solve):
    # 同一 update 周期内第二次 unscale_ 必须抛 RuntimeError
    out = solve()
    assert out["double_unscale_raises"] is True
