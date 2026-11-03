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
    noise=None,
    prediction_type="epsilon",
):
    """KDPM2-Ancestral：交错表 + σ_up/σ_down 噪声拆分的两调用状态机单步。

    返回 dict(prev_sample, pred_original_sample, state, derivative, dt,
    sigma_up, sigma_down)。

    ## 基础表与辅助表（每次调用重算，纯函数）

    入参 `sigmas` 是**未展开**的基础表：长度 N+1（N ≥ 2），[σ0, …, σ_{N−1}, 0]，
    有限、非负、单调不增，前 N 个 > 0、末位恰为 0.0。

    对基础表（官方 `set_timesteps`，k = 0…N）：
      σ_next = roll(−1)（σ_next[N] = 0）
      up[k]   = √( σ_next[k]²·(σ[k]² − σ_next[k]²) / σ[k]² )   ← k=N 处 0/0 = nan（官方同款，不被读）
      down[k] = √( σ_next[k]² − up[k]² )；再强制 down[N] = 0     ← k=N 处 nan 被清
      interpol[k] = exp( lerp(log σ[k], log down[k], 0.5) )；再强制 interpol[N−1] = interpol[N] = 0

    数学上 down[k] = σ_next[k]²/σ[k]、√(σ[k]·down[k]) = σ_next[k]——即 interpol 表
    精确数学下与 σ_next 逐元素相等（官方仍显式构造，浮点 float32 往返有 1e−7 级差）；
    `[-2:] = 0.0` 清场是修 k=N 处 (−inf) − (−inf) = nan 的 lerp 路径，顺带把本就是 0
    的 k=N−1 一并清掉。σ_up² + σ_down² = σ_next²：ancestral 把整步 σ→σ_next 拆成
    确定性部分 σ→σ_down 与噪声部分 σ_up。

    四张展开表（长度 2N+2，头 1 条 + 主体 repeat_interleave(2) + 尾 1 条）：
      sigE / sigIE / sigUp / sigDown 同构展开。

    ## 索引 → 阶段（末逻辑步只有首调）

    首调（state=None，step_index 偶数，0 ≤ i ≤ 2N−2）：逻辑步 j = i/2。
      σ = sigE[i]，m = sigIE[i]（interpol 中点；ancestral 的 interpol 表
      没有 [0,…] 死头，interpol[0] = √(σ0·down0) 本身就是逻辑步 0 的中点，
      展开后首调与次调读**同一格** sigIE[2j] = interp[j]）
      x0 = pred(sample, mo, σ)；derivative = (sample − x0)/σ
      dt = m − σ；prev = sample + derivative·dt（欧拉半步到中点）
      state = {"sample": sample}    ← 只存**原样本**
      官方首调还取出 sigmas_up[i]、sigmas_down[i−1]，但首调分支不读它们——死赋值
      （i=0 时 down[−1] 走负索引也不炸），本题直接不算。

    次调（state 为 dict，step_index 奇数，1 ≤ i ≤ 2N−3）：逻辑步 j = (i−1)/2。
      σ = sigE[i−1]，m = sigIE[i−1]，u = sigUp[i−1]，d = sigDown[i−1]
      x0 = pred(sample, mo, m)；derivative = (sample − x0)/m   ← 分母是中点
      dt = d − σ                                                ← **不是整步长**：噪声吃掉 up 部分
      prev = state["sample"] + derivative·dt + noise·u          ← 从**暂存样本**出发并注入噪声
      state = None；首调导数被丢弃

    与确定性版（diff-kdpm2-interleaved-step）的三点区别：① 次调 dt 用 σ_down−σ
    而非 σ_next−σ；② prev 落地后加 noise·σ_up；③ 中点插值对象是 σ_down 而非
    σ_next。末逻辑步 j=N−1：σ_next=0 → up=down=interpol=0，一步落到 x0 且无噪声。

    x0 的两种 prediction_type（sigma_input 首调用 σ、次调用 m）：
      epsilon      → sample − sigma_input·mo
      v_prediction → mo·(−sigma_input/√(sigma_input²+1)) + sample/(sigma_input²+1)

    noise：次调必需（官方由 generator 生成，本题改显式入参），形状须与
    model_output 一致；首调忽略。

    返回：prev_sample / pred_original_sample（float32）、state（dict 或 None）、
    derivative（float32）、dt（浮点数）、sigma_up / sigma_down（次调的浮点值，
    首调为 None）。
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

    # ---- 构造基础表与展开表（长度 2N+2）----
    sig_next = values.roll(-1)
    sig_next[-1] = 0.0
    sig_up = (sig_next**2 * (values**2 - sig_next**2) / values**2).sqrt()
    sig_down = (sig_next**2 - sig_up**2).sqrt()
    sig_down[-1] = 0.0
    interpol = values.log().lerp(sig_down.log(), 0.5).exp()
    interpol[-2:] = 0.0
    sig_e = torch.cat([values[:1], values[1:].repeat_interleave(2), values[-1:]])
    sig_ie = torch.cat([interpol[:1], interpol[1:].repeat_interleave(2), interpol[-1:]])
    sig_ue = torch.cat([sig_up[:1], sig_up[1:].repeat_interleave(2), sig_up[-1:]])
    sig_de = torch.cat([sig_down[:1], sig_down[1:].repeat_interleave(2), sig_down[-1:]])

    if state is None:
        # ---- 首调（欧拉半步到 interpol 中点，只暂存原样本）----
        if step_index % 2 != 0:
            raise ValueError("首调（state=None）的 step_index 必须为偶数")
        if step_index < 0 or step_index > 2 * num_steps - 2:
            raise ValueError(f"step_index 必须落在 [0, {2 * num_steps - 2}]")
        sigma = float(sig_e[step_index])
        sigma_interpol = float(sig_ie[step_index])
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
            "sigma_up": None,
            "sigma_down": None,
        }

    # ---- 次调（中点重估导数 + σ_down 步长 + σ_up 噪声注入）----
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
    if noise is None:
        raise ValueError("次调必须显式给 noise（官方由 generator 生成）")
    noise64 = _float_tensor("noise", noise)
    if noise64.shape != output64.shape:
        raise ValueError("noise 形状必须与 model_output 一致")

    sigma = float(sig_e[step_index - 1])
    sigma_interpol = float(sig_ie[step_index - 1])
    sigma_up = float(sig_ue[step_index - 1])
    sigma_down = float(sig_de[step_index - 1])
    if sigma <= 0.0:
        raise ValueError("次调的 sigma 必须为正")
    if sigma_interpol <= 0.0:
        raise ValueError("次调的中点 sigma 必须为正")
    x0 = _pred_x0(sample64, output64, sigma_interpol, prediction_type)
    derivative = (sample64 - x0) / sigma_interpol
    dt = sigma_down - sigma
    prev = stored + derivative * dt + noise64 * sigma_up
    return {
        "prev_sample": prev.to(torch.float32),
        "pred_original_sample": x0.to(torch.float32),
        "state": None,
        "derivative": derivative.to(torch.float32),
        "dt": dt,
        "sigma_up": sigma_up,
        "sigma_down": sigma_down,
    }
