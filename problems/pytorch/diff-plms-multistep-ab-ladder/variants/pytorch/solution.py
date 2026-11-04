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


def solve(
    model_output,
    sample,
    alphas_cumprod,
    timestep,
    step_ratio,
    counter,
    state=None,
    prediction_type="epsilon",
):
    """PLMS（PNDM 的 skip_prk_steps=True 全程路径）：ets 历史的 Adams-Bashforth
    阶梯 + cur_sample 重启的单调用状态机。

    返回 dict(prev_sample, model_output_combined, prev_timestep, state)。

    判题域钉死官方 Stable Diffusion 默认路径：skip_prk_steps = True（无 prk 子步），
    prediction_type 暴露 epsilon / v_prediction。`alphas_cumprod` 是官方训练侧
    ᾱ 表（一维浮点张量，长度 ≥ 2，全部 ∈ (0, 1]，单调不增）；`timestep` 是当前
    训练时间步索引；`step_ratio = num_train_timesteps // num_inference_steps ≥ 1`；
    `counter` 是全局调用计数（官方 self.counter，显式入参）。

    ## 状态机（官方 step_plms L324-392 逐字同构）

    state = {"ets": [e_θ 历史], "cur_sample": 上一步暂存样本或 None}。
    调用前契约：counter = 0 ⇔ state 为 None；counter ≥ 1 ⇔ state 给出且
    ets 长度恰为 [c1→1, c2→1, c3→2, c4→3, c≥5→4]（官方 append/pop 记账的
    自然形状——截断 [-3:] 发生在调用内，故调用前 c≥5 时持有 4 条）；
    cur_sample 仅在 counter = 1 时必须非 None（其余必须为 None）。

    1. **ets 记账**：counter ≠ 1 时 `ets = ets[-3:]` 截断再 append 当前输出
       （历史封顶 3 条旧值 + 1 条新值 = 4 条）；counter = 1 **不 append**。
    2. **组合输出**（按 append 后 ets 长度选 Adams-Bashforth 阶梯）：
       len 1 → 恒等（counter 0，同时暂存 cur_sample = sample）；
       len 1 且 counter = 1 → (mo + ets[-1]) / 2（两点梯形），且
       sample = state["cur_sample"]（**从 counter 0 暂存的样本重启**）；
       len 2 → (3·ets[-1] − ets[-2]) / 2；
       len 3 → (23·ets[-1] − 16·ets[-2] + 5·ets[-3]) / 12；
       len ≥ 4 → (55·ets[-1] − 59·ets[-2] + 37·ets[-3] − 9·ets[-4]) / 24。
    3. **timestep 记账**：counter 1 的 (timestep, prev_timestep) = (t+δ, t)
       （用 cur_sample 在同一段上重走整步），其余 = (t, t−δ)，δ = step_ratio。
    4. **落地 formula(9)**：α_t = alphas_cumprod[timestep]、
       α_prev = alphas_cumprod[prev_timestep]（**prev_timestep < 0 时取表头
       alphas_cumprod[0]，官方 final_alpha_cumprod**）；
       v_prediction 先把网络输出转回 ε：ε = √α_t·mo + √(1−α_t)·sample
       （sample 在 counter 1 已换成 cur_sample——转换发生在记账之后）；
       prev = √(α_prev/α_t)·sample − (α_prev − α_t)·ε /
       (α_t·√(1−α_prev) + √(α_t·(1−α_t)·α_prev))。

    counter = 1 组合出的输出与 counter = 0 的输出对同一对 (t+δ, t)——这是
    PNDM 论文「前两步 plms 等价于一次梯形步」的记账核心。末步 timestep = 0 →
    prev_timestep = −δ < 0 → 落到表头 ᾱ。

    校验失败一律抛 ValueError。state 各张量以 float64 存取。
    """
    output64 = _float_tensor("model_output", model_output)
    sample64 = _float_tensor("sample", sample)
    if sample64.shape != output64.shape:
        raise ValueError("sample 与 model_output 形状必须一致")

    if not torch.is_tensor(alphas_cumprod) or not alphas_cumprod.is_floating_point() or alphas_cumprod.dim() != 1:
        raise ValueError("alphas_cumprod 必须是一维浮点张量")
    alphas = alphas_cumprod.to(torch.float64)
    if alphas.numel() < 2:
        raise ValueError("alphas_cumprod 长度至少为 2")
    if not bool(torch.isfinite(alphas).all()):
        raise ValueError("alphas_cumprod 必须全部有限")
    if bool((alphas <= 0).any()) or bool((alphas > 1).any()):
        raise ValueError("alphas_cumprod 必须全部落在 (0, 1]")
    if bool((alphas[1:] > alphas[:-1]).any()):
        raise ValueError("alphas_cumprod 必须单调不增")

    if isinstance(timestep, bool) or not isinstance(timestep, int):
        raise ValueError("timestep 必须是整数")
    if timestep < 0 or timestep > alphas.numel() - 1:
        raise ValueError(f"timestep 必须落在 [0, {alphas.numel() - 1}]")

    if isinstance(step_ratio, bool) or not isinstance(step_ratio, int):
        raise ValueError("step_ratio 必须是整数")
    if step_ratio < 1:
        raise ValueError("step_ratio 必须至少为 1")

    if isinstance(counter, bool) or not isinstance(counter, int):
        raise ValueError("counter 必须是整数")
    if counter < 0:
        raise ValueError("counter 必须非负")

    if prediction_type not in _PREDICTION_TYPES:
        raise ValueError(f"未知的 prediction_type: {prediction_type!r}")

    # ---- 状态解析 ----
    if counter == 0:
        if state is not None:
            raise ValueError("counter 为 0 时必须传 state=None（冷启动）")
        ets = []
        cur_sample = None
    else:
        if not isinstance(state, dict) or set(state.keys()) != {"ets", "cur_sample"}:
            raise ValueError("state 必须是只含 ets / cur_sample 键的 dict")
        expected_len = 1 if counter in (1, 2) else (2 if counter == 3 else (3 if counter == 4 else 4))
        ets_raw = state["ets"]
        if not isinstance(ets_raw, list) or len(ets_raw) != expected_len:
            raise ValueError(f"state.ets 长度必须恰为 {expected_len}（counter={counter}）")
        ets = []
        for idx, item in enumerate(ets_raw):
            stored = _float_tensor(f"state.ets[{idx}]", item)
            if stored.shape != output64.shape:
                raise ValueError("state.ets 条目必须与 model_output 同形状")
            ets.append(stored)
        cur_sample = state["cur_sample"]
        if counter == 1:
            if cur_sample is None:
                raise ValueError("counter 为 1 时 state.cur_sample 必须存在（counter 0 暂存的样本）")
            cur_sample = _float_tensor("state.cur_sample", cur_sample)
            if cur_sample.shape != sample64.shape:
                raise ValueError("state.cur_sample 必须与 sample 同形状")
        elif cur_sample is not None:
            raise ValueError("counter ≥ 2 时 state.cur_sample 必须为 None")

    # ---- 1. ets 记账 ----
    if counter != 1:
        ets = ets[-3:]
        ets = ets + [output64]

    # ---- 2. AB 阶梯组合输出 ----
    combined = None
    if len(ets) == 1 and counter == 0:
        combined = output64
        cur_sample = sample64
    elif len(ets) == 1 and counter == 1:
        combined = (output64 + ets[-1]) / 2
        sample64 = cur_sample
        cur_sample = None
    elif len(ets) == 2:
        combined = (3 * ets[-1] - ets[-2]) / 2
    elif len(ets) == 3:
        combined = (23 * ets[-1] - 16 * ets[-2] + 5 * ets[-3]) / 12
    else:
        combined = (
            (55 * ets[-1] - 59 * ets[-2] + 37 * ets[-3] - 9 * ets[-4]) / 24
        )

    # ---- 3. timestep 记账 ----
    if counter == 1:
        prev_timestep = timestep
        timestep = timestep + step_ratio
    else:
        prev_timestep = timestep - step_ratio
    if timestep > alphas.numel() - 1:
        raise ValueError("counter 1 的 timestep + step_ratio 越过 alphas_cumprod 表尾")

    # ---- 4. 落地 formula(9) ----
    alpha_prod_t = float(alphas[timestep])
    alpha_prod_t_prev = (
        float(alphas[prev_timestep]) if prev_timestep >= 0 else float(alphas[0])
    )
    beta_prod_t = 1.0 - alpha_prod_t
    beta_prod_t_prev = 1.0 - alpha_prod_t_prev

    epsilon = combined
    if prediction_type == "v_prediction":
        epsilon = math.sqrt(alpha_prod_t) * combined + math.sqrt(beta_prod_t) * sample64

    sample_coeff = math.sqrt(alpha_prod_t_prev / alpha_prod_t)
    denom = alpha_prod_t * math.sqrt(beta_prod_t_prev) + math.sqrt(
        alpha_prod_t * beta_prod_t * alpha_prod_t_prev
    )
    prev = sample_coeff * sample64 - (alpha_prod_t_prev - alpha_prod_t) * epsilon / denom

    new_state = {
        "ets": ets,
        "cur_sample": cur_sample,
    }
    return {
        "prev_sample": prev.to(torch.float32),
        "model_output_combined": combined.to(torch.float32),
        "prev_timestep": prev_timestep,
        "state": new_state,
    }
