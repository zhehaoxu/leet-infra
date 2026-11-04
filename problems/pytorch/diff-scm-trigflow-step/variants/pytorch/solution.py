"""SCM (sCM / Sana-Sprint) trigflow 一致性步.

回源: diffusers ``schedulers/scheduling_scm.py`` (SCMScheduler.step, 305 行)。
sCM (Simplifying, Stabilizing & Scaling Continuous-Time Consistency Models,
arXiv 2410.11081) 的 trigflow 参数化把 timestep 定义为角度
theta = arctan(sigma / sigma_data) in [0, pi/2]:

    x(theta) = cos(theta) * x0 + sin(theta) * sigma_data * eps

一致性步的语义是「先跳回数据端、再重加噪到下一角度」(非 Markov):

    pred_x0      = cos(s) * sample - sin(s) * model_output
    prev_sample  = cos(t) * pred_x0 + sin(t) * (sigma_data * noise)

其中 s = timesteps[step_index] (当前角度), t = timesteps[step_index+1] (下一角度,
更干净)。t = 0 时 sin(t) = 0, 噪声项消失 -- Sana-Sprint 的单步采样就是
timesteps = [theta_max, 0] 的一步 (prev == pred_x0)。

与官方实现的判题域约定 (official -> judgement):
* 官方 step_index 状态机 (index_for_timestep / begin_index) 在判题域由显式
  step_index 入参代替; 官方支持重复 timestep 值的取位样板不入实现。
* 官方 randn_tensor 采样 (generator) 在管线侧, 判题域 noise 为显式入参
  (未缩放的原始噪声, 本函数内乘 sigma_data)。
* 官方 timesteps 由 linspace/set_timesteps 保证单调不增; 判题域显式校验,
  允许相邻相等 (官方 index_for_timestep 专门处理重复值)。
* noise=None 仅当 t 精确等于 0.0 时合法 (噪声项系数为零); t != 0 时必须
  显式给 noise。noise 非 None 但 t == 0.0 合法 -- 官方语义是采样了但乘零,
  prev 与 noise 无关。
* 官方 ``len(self.timesteps) > 1`` 门控的 else 分支 (prev = pred_x0) 只在
  num_inference_steps=0 的退化表下可达, 属死代码, 不入实现 -- 判题域表长
  >= 2, sin(t)=0 自然覆盖「末步不加噪」。

prediction_type 只支持官方唯一实现 "trigflow"。
"""

import math

import torch


def _as_scalar_angle(value, name):
    """从 1 元素/0 维张量或 Python 标量取 float 角度。"""
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"{name} 必须是标量, 得到 {value.numel()} 个元素")
        return float(value.reshape(()).item())
    raise ValueError(f"{name} 必须是 torch 张量, 得到 {type(value).__name__}")


def solve(model_output, sample, timesteps, step_index,
          noise=None, sigma_data=0.5, prediction_type="trigflow"):
    """SCM trigflow 一致性步。

    参数
    ----
    model_output : torch.Tensor
        模型输出 (trigflow 参数化), 与 sample 同形状。
    sample : torch.Tensor
        当前含噪样本 x(theta_s)。
    timesteps : torch.Tensor
        长度 N+1 (N >= 2) 的 1D 浮点角度表, 单调不增 (官方 linspace(max, 0, N+1)
        或 Sana-Sprint 的 [theta_max, 1.3, 0]); 相邻相等合法。
    step_index : int
        当前步索引, 须满足 0 <= step_index <= len(timesteps) - 2
        (要存在下一位角度 t)。
    noise : torch.Tensor or None
        原始噪声 (官方 randn_tensor 输出, 本函数内乘 sigma_data)。
        t == 0.0 时可省略 (系数为零); t != 0.0 时必须显式给出。
    sigma_data : float
        trigflow 数据端标准差 (官方默认 0.5), 必须 > 0。
    prediction_type : str
        只支持 "trigflow" (官方唯一实现)。

    返回
    ----
    dict(prev_sample, pred_original_sample, s_angle, t_angle, noise_term)
        * prev_sample        : cos(t)*pred_x0 + sin(t)*sigma_data*noise
        * pred_original_sample: cos(s)*sample - sin(s)*model_output
        * s_angle / t_angle  : 本步使用的两个角度 (float 标量)
        * noise_term         : sin(t)*sigma_data*noise 张量 (t=0 时为零张量;
                               noise=None 时为全零张量)
    """
    if not isinstance(model_output, torch.Tensor):
        raise ValueError("model_output 必须是 torch 张量")
    if not isinstance(sample, torch.Tensor):
        raise ValueError("sample 必须是 torch 张量")
    if model_output.shape != sample.shape:
        raise ValueError("model_output 与 sample 形状必须一致")
    if not isinstance(timesteps, torch.Tensor):
        raise ValueError("timesteps 必须是 torch 张量")
    if timesteps.dim() != 1:
        raise ValueError("timesteps 必须是一维张量")
    if timesteps.numel() < 2:
        raise ValueError("timesteps 表长必须 >= 2 (至少一对相邻角度)")
    if not (timesteps.dtype.is_floating_point):
        raise ValueError("timesteps 必须是浮点张量")
    diff = timesteps[1:] - timesteps[:-1]
    if bool((diff > 0).any()):
        raise ValueError("timesteps 必须单调不增 (官方 linspace(max, 0, N+1) 保证)")
    if isinstance(step_index, bool) or not isinstance(step_index, int):
        raise ValueError("step_index 必须是整数")
    if step_index < 0 or step_index > timesteps.numel() - 2:
        raise ValueError(
            f"step_index 必须落在 [0, {timesteps.numel() - 2}]"
        )
    if prediction_type != "trigflow":
        raise ValueError(f"Unsupported parameterization: {prediction_type}")
    if isinstance(sigma_data, bool) or not isinstance(sigma_data, (int, float)):
        raise ValueError("sigma_data 必须是数值")
    if sigma_data <= 0:
        raise ValueError("sigma_data 必须 > 0")

    s_angle = _as_scalar_angle(timesteps[step_index], "timesteps[step_index]")
    t_angle = _as_scalar_angle(timesteps[step_index + 1], "timesteps[step_index+1]")

    # trigflow: x(theta) = cos(theta)*x0 + sin(theta)*sigma_data*eps
    # => x0 = cos(s)*x - sin(s)*F(x) (官方唯一参数化)
    pred_original_sample = torch.cos(sample.new_tensor(s_angle)) * sample \
        - torch.sin(sample.new_tensor(s_angle)) * model_output

    noise_coef = math.sin(t_angle) * sigma_data
    if noise is None:
        if t_angle != 0.0:
            raise ValueError("t != 0 时必须显式给 noise (官方按 randn 采样)")
        noise_term = torch.zeros_like(pred_original_sample)
    else:
        if not isinstance(noise, torch.Tensor):
            raise ValueError("noise 必须是 torch 张量")
        if noise.shape != sample.shape:
            raise ValueError("noise 与 sample 形状必须一致")
        noise_term = noise_coef * noise

    prev_sample = torch.cos(sample.new_tensor(t_angle)) * pred_original_sample \
        + noise_term

    return {
        "prev_sample": prev_sample,
        "pred_original_sample": pred_original_sample,
        "s_angle": s_angle,
        "t_angle": t_angle,
        "noise_term": noise_term,
    }
