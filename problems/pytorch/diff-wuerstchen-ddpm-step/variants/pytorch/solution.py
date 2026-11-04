import math

import torch


def _float_tensor(name, value):
    if not torch.is_tensor(value):
        raise ValueError(f"{name} 必须是 torch 张量")
    if not value.is_floating_point():
        raise ValueError(f"{name} 必须是浮点张量")
    if value.numel() == 0:
        raise ValueError(f"{name} 不能为空张量")
    return value.to(torch.float64)


def _alpha_cumprod(t, scaler, s, init_ac):
    """官方 _alpha_cumprod 逐字：scaler 幂变换 t，再过带 s offset 的 cosine
    平方并除以 _init_alpha_cumprod 归一化，最后 clamp(0.0001, 0.9999)。"""
    if scaler > 1:
        t = 1 - (1 - t) ** scaler
    elif scaler < 1:
        t = t**scaler
    ac = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2 / init_ac
    return ac.clamp(0.0001, 0.9999)


def solve(
    model_output,
    timestep,
    sample,
    timesteps_table,
    noise=None,
    scaler=1.0,
    s=0.008,
):
    """DDPMWuerstchen：连续 t ∈ [0,1] 解析 cosine ᾱ + 查表 prev_t 的 DDPM
    后验单步（Wuerstchen 两级流水线的调度器，prior/decoder 各一套 scaler）。

    返回 dict(prev_sample, alpha_cumprod, alpha_cumprod_prev, alpha_ratio,
    mu, noise_applied, prev_t)。

    与离散 ᾱ 表的 DDPM（diff-ddpm-step）不同，本题 ᾱ 是 **t 的解析函数**：

    - `_init_alpha_cumprod` = cos(s/(1+s)·π/2)²（s 默认 0.008）；
    - scaler 幂变换：>1 → t = 1−(1−t)^scaler；<1 → t = t^scaler；=1 恒等；
    - ᾱ(t) = cos((t+s)/(1+s)·π/2)² / _init_alpha_cumprod，
      clamp 到 [0.0001, 0.9999]（t=1 → 0 → 下界；t=0 → 1 → 上界）。

    ## 查表（官方 previous_timestep 逐字）

    index = argmin(|timesteps_table − timestep[0]|)（平局取首个，torch 语义），
    prev_t = timesteps_table[index + 1]——**官方只用 timestep[0] 定位**且
    prev_t 对整个 batch 广播；index+1 越界（timestep 已是表尾、无后继）拒绝。
    timesteps_table 须一维、长度 ≥ 2、有限、单调不增（官方 linspace(1,0,N+1)）。

    ## 后验一步（官方 step 逐字，batch 维广播）

    ᾱ_t = ᾱ(timestep)、ᾱ_prev = ᾱ(prev_t)（各 view 成 (B, 1, …)）、
    α = ᾱ_t/ᾱ_prev；

    - mu = √(1/α)·(sample − (1−α)·model_output/√(1−ᾱ_t))；
    - std = √((1−α)·(1−ᾱ_prev)/(1−ᾱ_t))·noise；
    - **门控**：prev = mu + std·(prev_t != 0)——prev_t 精确为 0（末步）时
      不加噪声（DDIM 式确定性收尾），浮点比较；
    - ᾱ_t = ᾱ_prev 时 α = 1 → mu = sample、std = 0 → prev = sample（恒等步）。

    timestep 须一维浮点张量、有限、值域 [0, 1]、**batch 内全部相等**（官方
    管线每步共享一个 t；官方只用 t[0] 查 prev_t，混合 batch 语义破坏故拒绝）；
    sample / model_output / noise 同形状且 batch 维 = len(timestep)；noise 每次
    调用必需（官方 randn_tensor）。scaler 须正有限；s 须正有限。

    校验失败一律抛 ValueError。
    """
    output64 = _float_tensor("model_output", model_output)
    sample64 = _float_tensor("sample", sample)
    if sample64.shape != output64.shape:
        raise ValueError("sample 与 model_output 形状必须一致")

    t = _float_tensor("timestep", timestep)
    if t.dim() != 1:
        raise ValueError("timestep 必须是一维张量（batch 维）")
    if not bool(torch.isfinite(t).all()):
        raise ValueError("timestep 必须全部有限")
    if bool((t < 0).any()) or bool((t > 1).any()):
        raise ValueError("timestep 必须落在 [0, 1]")
    if t.numel() > 1 and bool((t != t[0]).any()):
        raise ValueError("batch 内 timestep 必须全部相等（官方每步共享一个 t）")

    table = _float_tensor("timesteps_table", timesteps_table)
    if table.dim() != 1:
        raise ValueError("timesteps_table 必须是一维张量")
    if table.numel() < 2:
        raise ValueError("timesteps_table 长度至少为 2")
    if not bool(torch.isfinite(table).all()):
        raise ValueError("timesteps_table 必须全部有限")
    if bool((table[1:] > table[:-1]).any()):
        raise ValueError("timesteps_table 必须单调不增（官方 linspace(1, 0, N+1)）")

    if isinstance(scaler, bool) or not isinstance(scaler, (int, float)):
        raise ValueError("scaler 必须是数值")
    scaler = float(scaler)
    if not math.isfinite(scaler) or scaler <= 0:
        raise ValueError("scaler 必须为正有限")
    if isinstance(s, bool) or not isinstance(s, (int, float)):
        raise ValueError("s 必须是数值")
    s = float(s)
    if not math.isfinite(s) or s <= 0:
        raise ValueError("s 必须为正有限")

    if noise is None:
        raise ValueError("必须显式给 noise（官方 randn_tensor 每步生成）")
    noise64 = _float_tensor("noise", noise)
    if noise64.shape != sample64.shape:
        raise ValueError("noise 形状必须与 sample 一致")

    # ---- 查表 prev_t（官方 previous_timestep：argmin 最近邻 + 1）----
    index = int(torch.argmin(torch.abs(table - t[0])).item())
    if index + 1 >= table.numel():
        raise ValueError("timestep 已是 timesteps_table 表尾，无后继步")
    prev_t = float(table[index + 1].item())

    batch = t.numel()
    view = [batch] + [1] * (sample64.dim() - 1)

    init_ac = math.cos(s / (1 + s) * math.pi * 0.5) ** 2
    ac = _alpha_cumprod(t, scaler, s, init_ac).view(view)
    acp = _alpha_cumprod(torch.tensor([prev_t], dtype=torch.float64), scaler, s, init_ac)
    acp = acp.expand(batch).view(view)
    alpha = ac / acp

    mu = (1.0 / alpha).sqrt() * (sample64 - (1 - alpha) * output64 / (1 - ac).sqrt())
    std = ((1 - alpha) * (1.0 - acp) / (1.0 - ac)).sqrt() * noise64
    gate = 1.0 if prev_t != 0.0 else 0.0
    prev = mu + std * gate

    return {
        "prev_sample": prev.to(torch.float32),
        "alpha_cumprod": ac.reshape(batch).to(torch.float32),
        "alpha_cumprod_prev": acp.reshape(batch).to(torch.float32),
        "alpha_ratio": alpha.reshape(batch).to(torch.float32),
        "mu": mu.to(torch.float32),
        "noise_applied": prev_t != 0.0,
        "prev_t": prev_t,
    }
