def solve(
    pred_cond,
    pred_uncond,
    pred_cond_skip,
    guidance_scale,
    perturbed_scale,
    step,
    num_inference_steps,
    start=0.0,
    stop=1.0,
    perturbed_start=0.01,
    perturbed_stop=0.2,
):
    """PAG 双尺度引导的四种分支。

    三段预测同形状（第 0 维为 batch）：
      pred_cond        —— 条件预测；
      pred_uncond      —— 无条件预测，配合 guidance_scale 做 CFG；
      pred_cond_skip   —— 扰动注意力分支的条件预测，配合 perturbed_scale 做 PAG。

    step / num_inference_steps 决定两套窗口：
      CFG 窗口为左闭右开的整数区间 [int(start*N), int(stop*N))；
      PAG 窗口为开区间 (int(perturbed_start*N), int(perturbed_stop*N))。
    另外 guidance_scale 与 1.0 相近、perturbed_scale 与 0.0 相近时，对应分支视为关闭。

    四个分支（与 diffusers 一致）：
      两者关闭            -> pred_cond
      仅 PAG 开启         -> pred_cond_skip + perturbed_scale * (pred_cond - pred_cond_skip)
      仅 CFG 开启         -> pred_uncond + guidance_scale * (pred_cond - pred_uncond)
      两者开启            -> pred_uncond + guidance_scale * (pred_cond - pred_uncond)
                                        + perturbed_scale * (pred_cond - pred_cond_skip)

    某个分支关闭时对应的输入可以为 None；但若该分支生效却传 None，抛 ValueError。
    返回 float32 张量，形状与 pred_cond 相同。
    """
    # TODO
    raise NotImplementedError
