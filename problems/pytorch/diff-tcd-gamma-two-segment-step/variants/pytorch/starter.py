"""TCD gamma 两段式一致性步 -- 待实现骨架.

回源: diffusers ``schedulers/scheduling_tcd.py`` (TCDScheduler.step)。
TCD 的核心: gamma 中间噪声级 + 两段式采样。

    timestep_s = floor((1 - eta) * prev_timestep)      # prev_t = timesteps[step_index+1] 或表尾越界取 0
    # 参数化 (epsilon 为例):
    x0 = (x_t - sqrt(1-abar_t)*eps) / sqrt(abar_t)
    x_s = sqrt(abar_s)*x0 + sqrt(1-abar_s)*eps          # 第一段: 跳到中间噪声级
    # 第二段 (eta>0 且非末步):
    prev = sqrt(abar_prev/abar_s)*x_s + sqrt(1-abar_prev/abar_s)*noise
    # 末步 (step_index == len(timesteps)-1) 或 eta==0: prev = x_s

请实现 solve:
* 校验 mo/sample 类型与形状; alphas_cumprod 为 1D 浮点张量长度>=2;
  timesteps 为 1D 整型、严格递减、元素在 ᾱ 索引范围内;
* step_index 为 int (bool 拒绝) 且 0 <= step_index <= len(timesteps)-1;
* eta 为数值且 ∈ [0,1]; prediction_type ∈ {epsilon, sample, v_prediction};
* eta>0 且非末步时 noise 必须显式给 (类型/形状), 否则忽略;
* 返回 dict(prev_sample, pred_noised_sample, pred_original_sample, pred_epsilon,
  timestep_s, prev_timestep, noise_applied)。
"""

import torch


def solve(model_output, sample, alphas_cumprod, timesteps, step_index,
          noise=None, eta=0.3, prediction_type="epsilon"):
    raise NotImplementedError
