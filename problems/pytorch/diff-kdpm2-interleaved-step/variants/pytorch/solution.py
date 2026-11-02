import math

import torch

_PREDICTION_TYPES = ("epsilon", "v_prediction")


def _float_tensor(name, value):
    if not torch.is_tensor(value):
        raise ValueError(f"{name} 必须是 torch 张量")
    if not value.is_floating_point():
        raise ValueError(f"{name} 必须是浮点张量")
    if value.numel() == 0:
        raise ValueError(f"{name} 不能为空张量")
    return value.to(torch.float64)


def _pred_x0(sample64, out64, sigma_input, prediction_type):
    if prediction_type == "epsilon":
        return sample64 - sigma_input * out64
    return out64 * (-sigma_input / math.sqrt(sigma_input**2 + 1.0)) + sample64 / (sigma_input**2 + 1.0)


def solve(
    model_output,
    sample,
    sigmas,
    step_index,
    state=None,
    prediction_type="epsilon",
):
    """KDPM2（确定性 DPM-Solver-2）交错表上的两调用状态机单步。

    返回 dict(prev_sample, pred_original_sample, state, derivative, dt)。

    ## 交错表的构造（每次调用重算，纯函数）

    入参 `sigmas` 是**未展开**的基础表：长度 N+1（N ≥ 2），[σ0, …, σ_{N−1}, 0]，
    有限、非负、单调不增，且前 N 个 > 0、末位恰为 0.0。

    中点表（长度 N+1）：I[k] = √(σ[k−1]·σ[k])（k = 1…N），I[0] = 0。
    与官方 `sigmas.log().lerp(sigmas.roll(1).log(), 0.5).exp()` 数值同构
    （√(ab) = exp((log a + log b)/2)）。注意官方 torch 路径下 I[0] = exp(nan) = nan、
    I[N] = exp(−inf) = 0（numpy 路径反之：I[0] = 0、I[N] = nan）——两个边界项
    **都不会被 step 读到**，ancestral 版干脆 `sigmas_interpol[-2:] = 0.0` 清场。

    两张展开表（长度 2N+2，头 1 条 + 主体 repeat_interleave(2) + 尾 1 条）：
      sigE = [σ0, σ1,σ1, σ2,σ2, …, σ_{N−1},σ_{N−1}, 0, 0, 0]
      sigIE = [0, I1,I1, I2,I2, …, I_{N−1},I_{N−1}, 0, 0, 0]
    即 sigE[2k] = σ_k（0 ≤ k ≤ N），sigE[2k+1] = σ_{k+1}（0 ≤ k ≤ N−1），
    sigE[2N+1] = 0（尾部多补的一条）；sigIE 同构。

    ## 索引 → 阶段（timesteps 长 2N−1，最后一个逻辑步**只有首调**）

    首调（state=None，step_index 偶数，0 ≤ i ≤ 2N−2）：逻辑步 j = i/2。
      σ = sigE[i] = σ_j（σ_hat = σ·(gamma+1)，官方 gamma=0 恒等），中点 m = sigIE[i+1] = I_{j+1}。
      x0 = pred(sample, mo, σ)
      derivative = (sample − x0) / σ          ← 分母是当前 σ
      dt = m − σ
      prev = sample + derivative·dt           ← 欧拉半步走到中点
      state = {"sample": sample}              ← 只存**原样本**（中点前的那个）

    次调（state 为 dict，step_index 奇数，1 ≤ i ≤ 2N−3）：逻辑步 j = (i−1)/2，
    末逻辑步（j = N−1）没有次调。sample 是 pipeline 在中点上的模型输入，mo 是其输出。
      σ = sigE[i−1] = σ_j，m = sigIE[i] = I_{j+1}（必须 > 0），σ_next = sigE[i] = σ_{j+1}
      x0 = pred(sample, mo, m)                ← sigma_input 用**中点**
      derivative = (sample − x0) / m          ← 分母是中点
      dt = σ_next − σ                         ← **整步长**（不是半步）
      prev = state["sample"] + derivative·dt  ← 从**暂存样本**出发；首调的导数被丢弃
      state = None

    与 Heun 的三点区别：① 首调导数不参与平均（DPM-Solver-2 只在中点重估一次）；
    ② 中点是 log-σ 几何平均而非"步长减半"；③ 次调的 dt 是从 σ_j 到 σ_{j+1} 的整步，
    分母是中点 m 而不是 σ_{j+1}。

    x0 的两种 prediction_type（sigma_input 首调用 σ、次调用 m）：
      epsilon      → sample − sigma_input·mo
      v_prediction → mo·(−sigma_input/√(sigma_input²+1)) + sample/(sigma_input²+1)

    返回：prev_sample / pred_original_sample（float32）、state（dict 或 None）、
    derivative（本调用算得的导数，float32）、dt（浮点数）。
    """
    output64 = _float_tensor("model_output", model_output)
    sample64 = _float_tensor("sample", sample)
    if sample64.shape != output64.shape:
        raise ValueError("sample 与 model_output 形状必须一致")

    if not torch.is_tensor(sigmas) or not sigmas.is_floating_point() or sigmas.dim() != 1:
        raise ValueError("sigmas 必须是一维浮点张量")
    values = sigmas.to(torch.float64)
    if values.numel() < 3:
        raise ValueError("sigmas 长度至少为 3（N ≥ 2 个非零 σ + 末位 0）")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("sigmas 必须全部有限")
    if bool((values < 0).any()):
        raise ValueError("sigmas 必须非负")
    if bool((values[1:] > values[:-1]).any()):
        raise ValueError("sigmas 必须单调不增")
    if float(values[-1]) != 0.0:
        raise ValueError("sigmas 末位必须恰为 0（官方 set_timesteps 会追加）")
    if bool((values[:-1] <= 0).any()):
        raise ValueError("sigmas 前 N 个必须为正")

    num_steps = values.numel() - 1  # N
    if isinstance(step_index, bool) or not isinstance(step_index, int):
        raise ValueError("step_index 必须是整数")

    if prediction_type not in _PREDICTION_TYPES:
        raise ValueError(f"未知的 prediction_type: {prediction_type!r}")

    # ---- 构造展开表（长度 2N+2）----
    prod = (values[:-1] * values[1:]).sqrt()  # [I1 … I_N]，I_N = 0
    interpol = torch.cat([torch.zeros(1, dtype=torch.float64), prod])  # [0, I1 … I_N]
    sig_e = torch.cat([values[:1], values[1:].repeat_interleave(2), values[-1:]])
    sig_ie = torch.cat([interpol[:1], interpol[1:].repeat_interleave(2), interpol[-1:]])

    if state is None:
        # ---- 首调（欧拉半步到中点）----
        if step_index % 2 != 0:
            raise ValueError("首调（state=None）的 step_index 必须为偶数")
        if step_index < 0 or step_index > 2 * num_steps - 2:
            raise ValueError(f"step_index 必须落在 [0, {2 * num_steps - 2}]")
        sigma = float(sig_e[step_index])
        sigma_interpol = float(sig_ie[step_index + 1])
        x0 = _pred_x0(sample64, output64, sigma, prediction_type)
        derivative = (sample64 - x0) / sigma
        dt = sigma_interpol - sigma
        prev = sample64 + derivative * dt
        return {
            "prev_sample": prev.to(torch.float32),
            "pred_original_sample": x0.to(torch.float32),
            "state": {"sample": sample64},
            "derivative": derivative.to(torch.float32),
            "dt": dt,
        }

    # ---- 次调（DPM-Solver-2 整步更新）----
    if not isinstance(state, dict) or set(state.keys()) != {"sample"}:
        raise ValueError("state 必须是只含 sample 键的 dict")
    if step_index % 2 != 1:
        raise ValueError("次调的 step_index 必须为奇数")
    if step_index < 1 or step_index > 2 * num_steps - 3:
        raise ValueError(
            f"次调的 step_index 必须落在 [1, {2 * num_steps - 3}]（末逻辑步只有首调）"
        )
    stored = _float_tensor("state.sample", state["sample"])
    if stored.shape != sample64.shape:
        raise ValueError("state.sample 必须与 sample 同形状")

    sigma = float(sig_e[step_index - 1])
    sigma_interpol = float(sig_ie[step_index])
    sigma_next = float(sig_e[step_index])
    if sigma <= 0.0:
        raise ValueError("次调的 sigma 必须为正")
    if sigma_interpol <= 0.0:
        raise ValueError("次调的中点 sigma 必须为正")
    x0 = _pred_x0(sample64, output64, sigma_interpol, prediction_type)
    derivative = (sample64 - x0) / sigma_interpol
    dt = sigma_next - sigma
    prev = stored + derivative * dt
    return {
        "prev_sample": prev.to(torch.float32),
        "pred_original_sample": x0.to(torch.float32),
        "state": None,
        "derivative": derivative.to(torch.float32),
        "dt": dt,
    }
