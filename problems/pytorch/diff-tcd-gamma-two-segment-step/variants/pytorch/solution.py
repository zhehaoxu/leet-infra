"""TCD (Trajectory Consistency Distillation) gamma 两段式一致性步.

回源: diffusers ``schedulers/scheduling_tcd.py`` (TCDScheduler.step, 810 行)。
TCD (arXiv 2410.05964) 相对 LCM 的核心增量是 gamma 中间噪声级与两段式采样:

    timestep_s = floor((1 - eta) * prev_timestep)        # gamma 插值 (论文记 gamma, 代码记 eta)
    x0  = (x_t - sqrt(1-abar_t) * eps) / sqrt(abar_t)    # epsilon 参数化
    x_s = sqrt(abar_s) * x0 + sqrt(1 - abar_s) * eps     # 先跳到中间噪声级 s
    prev = sqrt(abar_prev/abar_s) * x_s
         + sqrt(1 - abar_prev/abar_s) * noise            # 再加噪到 prev (eta>0 且非末步)

末步 (step_index == len(timesteps)-1) 或 eta == 0 时 prev = x_s (不加噪)。
注意 prev_timestep 取 timesteps[step_index+1], 越过表尾时取 0 (官方 torch.tensor(0))。

与官方实现的判题域约定 (official -> judgement):
* 官方 alphas_cumprod[timestep] 用 timestep 直接索引训练表; 判题域 timestep 由
  step_index 从 timesteps 表定位 (官方 self.timesteps), timesteps 元素必须是
  alphas_cumprod 的合法索引 (官方越界会 IndexError, 判题域显式拒绝)。
* 官方自定义 timesteps 路径校验严格递减 (`timesteps[i] >= timesteps[i-1]` 报错);
  判题域对所有表执行同一校验 (官方线性构造天然严格递减)。
* 官方 eta 用 assert (AssertionError); 判题域统一 ValueError, 消息保留
  "gamma must be less than or equal to 1.0" 关键词。
* 官方 `alpha_prod_t_prev = alphas_cumprod[prev_timestep] if prev_timestep >= 0
  else final_alpha_cumprod` 的 else 分支不可达 (prev_timestep 恒 >= 0), 死代码
  不入实现; 末步 prev_timestep=0 直接索引 alphas_cumprod[0]。
* 官方 randn_tensor (generator) 在管线侧, 判题域 noise 显式入参; 需要 noise 的
  条件是 eta > 0 且 step_index != len(timesteps)-1, 此时必须显式给; 不需要时
  (eta=0 或末步) 即使传了也忽略 (官方语义: 分支不走, 采样不发生)。
"""

import math

import torch


def solve(model_output, sample, alphas_cumprod, timesteps, step_index,
          noise=None, eta=0.3, prediction_type="epsilon"):
    """TCD gamma 两段式一致性步。

    参数
    ----
    model_output : torch.Tensor
        模型输出 (prediction_type 决定语义), 与 sample 同形状。
    sample : torch.Tensor
        当前含噪样本 x_t。
    alphas_cumprod : torch.Tensor
        训练表 ᾱ (1D 浮点, 长度 >= 2), timestep/prev_timestep/timestep_s 是它的索引。
    timesteps : torch.Tensor
        推理表 (1D 整型, 严格递减, 元素 ∈ [0, len(ᾱ)-1]); 官方 num_inference_steps
        = len(timesteps)。允许长度 1 (单步采样)。
    step_index : int
        当前步索引, 0 <= step_index <= len(timesteps)-1。
    noise : torch.Tensor or None
        原始噪声 (官方 randn_tensor 输出)。eta > 0 且非末步时必须显式给;
        eta = 0 或末步时忽略。
    eta : float
        随机性参数 (论文的 gamma), ∈ [0, 1]; 0 确定性、1 全随机。
    prediction_type : str
        "epsilon" | "sample" | "v_prediction"。

    返回
    ----
    dict(prev_sample, pred_noised_sample, pred_original_sample, pred_epsilon,
         timestep_s, prev_timestep, noise_applied)
        * prev_sample          : 最终样本
        * pred_noised_sample   : 中间噪声级 x_s
        * pred_original_sample : x0
        * pred_epsilon         : 预测噪声 ε
        * timestep_s           : floor((1-eta)*prev_t) (int)
        * prev_timestep        : timesteps[step_index+1] 或 0 (int)
        * noise_applied        : 是否注入了噪声 (bool)
    """
    if not isinstance(model_output, torch.Tensor):
        raise ValueError("model_output 必须是 torch 张量")
    if not isinstance(sample, torch.Tensor):
        raise ValueError("sample 必须是 torch 张量")
    if model_output.shape != sample.shape:
        raise ValueError("model_output 与 sample 形状必须一致")
    if not isinstance(alphas_cumprod, torch.Tensor):
        raise ValueError("alphas_cumprod 必须是 torch 张量")
    if alphas_cumprod.dim() != 1:
        raise ValueError("alphas_cumprod 必须是一维张量")
    if alphas_cumprod.numel() < 2:
        raise ValueError("alphas_cumprod 长度必须 >= 2")
    if not alphas_cumprod.dtype.is_floating_point:
        raise ValueError("alphas_cumprod 必须是浮点张量")
    if not isinstance(timesteps, torch.Tensor):
        raise ValueError("timesteps 必须是 torch 张量")
    if timesteps.dim() != 1:
        raise ValueError("timesteps 必须是一维张量")
    if timesteps.numel() < 1:
        raise ValueError("timesteps 不能为空")
    if timesteps.dtype not in (torch.int64, torch.int32):
        raise ValueError("timesteps 必须是整型张量")
    if int(timesteps.max()) > alphas_cumprod.numel() - 1 or int(timesteps.min()) < 0:
        raise ValueError("timesteps 元素必须落在 alphas_cumprod 的索引范围内")
    diff = timesteps[1:] - timesteps[:-1]
    if timesteps.numel() > 1 and bool((diff >= 0).any()):
        raise ValueError("timesteps 必须严格递减 (官方 custom_timesteps 校验)")
    if isinstance(step_index, bool) or not isinstance(step_index, int):
        raise ValueError("step_index 必须是整数")
    if step_index < 0 or step_index > timesteps.numel() - 1:
        raise ValueError(f"step_index 必须落在 [0, {timesteps.numel() - 1}]")
    if isinstance(eta, bool) or not isinstance(eta, (int, float)):
        raise ValueError("eta 必须是数值")
    if eta < 0 or eta > 1.0:
        raise ValueError("gamma must be less than or equal to 1.0")
    if prediction_type not in ("epsilon", "sample", "v_prediction"):
        raise ValueError(
            f"prediction_type given as {prediction_type} must be one of `epsilon`, `sample` or"
            " `v_prediction` for `TCDScheduler`."
        )

    timestep = int(timesteps[step_index].item())
    prev_step_index = step_index + 1
    if prev_step_index < timesteps.numel():
        prev_timestep = int(timesteps[prev_step_index].item())
    else:
        prev_timestep = 0
    timestep_s = math.floor((1.0 - eta) * prev_timestep)

    alpha_prod_t = float(alphas_cumprod[timestep].item())
    beta_prod_t = 1.0 - alpha_prod_t
    alpha_prod_t_prev = float(alphas_cumprod[prev_timestep].item())
    alpha_prod_s = float(alphas_cumprod[timestep_s].item())
    beta_prod_s = 1.0 - alpha_prod_s

    if prediction_type == "epsilon":
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_epsilon = model_output
    elif prediction_type == "sample":
        pred_original_sample = model_output
        pred_epsilon = (sample - alpha_prod_t ** 0.5 * pred_original_sample) / beta_prod_t ** 0.5
    else:  # v_prediction
        pred_original_sample = alpha_prod_t ** 0.5 * sample - beta_prod_t ** 0.5 * model_output
        pred_epsilon = alpha_prod_t ** 0.5 * model_output + beta_prod_t ** 0.5 * sample

    # 两段式的第一段: 跳到中间噪声级 s (官方三参数化共用)
    pred_noised_sample = alpha_prod_s ** 0.5 * pred_original_sample + beta_prod_s ** 0.5 * pred_epsilon

    # 第二段: eta > 0 且非末步时加噪到 prev
    noise_applied = eta > 0 and step_index != timesteps.numel() - 1
    if noise_applied:
        if noise is None:
            raise ValueError("eta > 0 且非末步时必须显式给 noise (官方按 randn 采样)")
        if not isinstance(noise, torch.Tensor):
            raise ValueError("noise 必须是 torch 张量")
        if noise.shape != sample.shape:
            raise ValueError("noise 与 sample 形状必须一致")
        ratio = alpha_prod_t_prev / alpha_prod_s
        prev_sample = ratio ** 0.5 * pred_noised_sample + (1.0 - ratio) ** 0.5 * noise
    else:
        prev_sample = pred_noised_sample

    return {
        "prev_sample": prev_sample,
        "pred_noised_sample": pred_noised_sample,
        "pred_original_sample": pred_original_sample,
        "pred_epsilon": pred_epsilon,
        "timestep_s": timestep_s,
        "prev_timestep": prev_timestep,
        "noise_applied": noise_applied,
    }
