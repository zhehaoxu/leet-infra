def solve(
    pred_cond,
    pred_uncond,
    step,
    num_inference_steps,
    buffer=None,
    cfg_scale=3.5,
    apg_scale=10.0,
    momentum=-0.5,
    norm_threshold=10.0,
    eta=0.0,
    apg_start_step=5,
    start=0.0,
    stop=1.0,
    use_original_formulation=False,
):
    """APG-Mix：先跑 CFG 预热动量，再切到投影引导。

    两段同形预测（第 0 维为 batch）：pred_cond 条件预测、pred_uncond 无条件预测。
    buffer 是跨步携带的动量状态（同形张量），首次调用传 None 表示从 0 起步；
    momentum 为 None 时不做动量，此时 buffer 必须是 None。

    先判两件事（N = num_inference_steps，s_cfg = cfg_scale，s_apg = apg_scale）：
      cfg_on = [int(start*N) <= step < int(stop*N)] and not isclose(s_cfg, 1.0)
      apg_on = cfg_on and step > apg_start_step and not isclose(s_apg, 1.0)

    三个分支：
      not cfg_on -> pred = pred_cond，buffer 原样带回（不更新）
      cfg_on 且 not apg_on -> 先用 d = pred_cond - pred_uncond 更新动量（若有），
                              再 pred = pred_uncond + s_cfg * d
      apg_on -> 用 d 更新动量后取 d = 动量值（无动量时 d 就是 pred_cond - pred_uncond），
                对 d 做范数截断（norm_threshold > 0 时乘 min(1, norm_threshold/‖d‖₂)，
                范数在除 batch 外的所有维度上求），
                再把 d 分解为沿 pred_cond 单位方向的分量与正交分量：
                  u = pred_cond / ‖pred_cond‖
                  d_parallel = <d, u> * u， d_orth = d - d_parallel
                取 normalized = d_orth + eta * d_parallel，
                pred = pred_uncond + s_apg * normalized

    动量更新公式：running = d + momentum * running_prev（running_prev 为 0 表示还未初始化）。

    use_original_formulation=True 时基准预测从 pred_uncond 换成 pred_cond，且「尺度退化」
    的判据也从 isclose(s, 1.0) 换成 isclose(s, 0.0)（CFG 与 APG 两个尺度都如此）。
    返回 (pred, new_buffer)：均为 float32，new_buffer 与 pred_cond 同形；
    无动量时 new_buffer 为 None。输入 buffer 不被就地修改。
    """
    # TODO
    raise NotImplementedError
