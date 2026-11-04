# 扩散：FasterCache CFG-Cache 频域逼近步（delta 状态机 + 低频掩码重建）

## 题意

FasterCache 的 denoiser 侧 CFG-Cache：CFG 采样中每个时间步都要跑 uncond + cond 两份前向，但相邻步的 `uncond − cond` 输出差在**频域**上缓变——用上一步的频域 delta 逼近本步 uncond 分支，省掉一半 batch 前向。实现 `solve(model_output, state, current_timestep, unconditional_batch_skip_range=5, unconditional_batch_timestep_skip_range=(-1, 641), low_frequency_weight_update_timestep_range=(99, 901), high_frequency_weight_update_timestep_range=(-1, 301), alpha_low_frequency=1.1, alpha_high_frequency=1.1, is_guidance_distilled=False)`：

```text
# 1. 门控 (开区间 + 取模)
within = lo < current_timestep < hi                  # 开区间 (严格)
skip_uncond = within and iteration % N != 0 and not distilled

# 2. 非 skip 步: 整批前向 (输出 2B = uncond||cond)
uncond, cond = chunk(2, dim=0)
各自 BCFHW -> permute(0,2,1,3,4) -> flatten(0,1) 得 (B*F, C, H, W)
_split_low_high_freq(x):
    fft2 -> fftshift (全维) -> 圆形低频掩码
    radius = min(H, W) // 5, 中心 (W//2, H//2)
    low = shifted * mask, high = shifted * ~mask
delta_low  = low(uncond)  - low(cond)                # 复数张量, 存入 state
delta_high = high(uncond) - high(cond)

# 3. skip 步: 只跑 cond (输出 B)
weight_low  = alpha_low  if lo_l < t < hi_l else 1.0 # 各自更新开区间
weight_high = alpha_high if lo_h < t < hi_h else 1.0
delta_low, delta_high <- 各自 * weight               # 作用在**存储的** delta 上
uncond_freq = (delta_low + low(cond)) + (delta_high + high(cond))
uncond = real(ifft2(ifftshift(uncond_freq)))         # 还原 BCFHW 排布
output = cat([uncond, cond], dim=0)                  # 2B
skip 步 state 存**缩放后**的 delta; 非 skip 步重算覆盖
```

返回 `dict(output, uncond_skipped, state)`，state 四键 `{iteration, batch_size, low_frequency_delta, high_frequency_delta}`（新 dict，纯函数式；batch_size 在首个非 skip 步记录并保持，正偶数）。

## 关键语义

- **门控三条件缺一不可**：时间步开区间、`iteration % N != 0`（第 0, N, 2N… 步真算整批）、非蒸馏。蒸馏模型恒整批前向，delta 永不建立。
- **skip 步输入是条件半批（B）**：`batch * 2 == state.batch_size` 必须成立，且 delta 必须已建立（前置整批步先更新）——首步直接 skip 会显式报错。
- **权重作用在存储的 delta 上**：`delta <- delta * weight` 是对 state 的累积缩放（官方 Equation 11），连续 skip 会指数叠加（×alpha^k）；整批步重算覆盖而非继续缩放。
- **低频掩码是圆形严格包含**：`(x−cx)² + (y−cy)² <= r²`，r = min(H,W)//5，fftshift 后中心在 (W//2, H//2)；low/high 支撑互补。
- **重建管道不可省一步**：ifftshift 还原频谱排布 → ifft2 → 取实部 → unflatten + permute 还原 BCFHW → cat 时 uncond 在前。
- **官方 `iteration > 0 and ...` 的 `>0` 是死代码**（0%N==0 恒真），判题域不收入。

## 判题域约定

- 官方 hook 包住 denoiser 前向、权重由 callback 提供；判题域显式入参，权重按官方缺省规则内联（alpha 在各自更新开区间内生效，区间外 1.0）。
- `model_output` 是「本次实际执行的输出」：skip 步为条件半批（B），否则整批（2B）。
- tensor_format 固定 "BCFHW"（官方 BFCHW/BCHW 分支与 tuple 输出分支不入）。
- 官方不校验；判题域补张量/区间/alpha/state 四键/skip 前置（batch_size 与 delta 已建立）/输入 batch 与门控一致等显式校验。

## 例子

非蒸馏完整序列（整批 2B=4、条件半批 B=2、N=5、t=400 在区间内）：iter0 整批 compute（delta = d0）；iter1-4 条件半批 skip（uncond 由 `(d0·w + low_cond) + (d0·w + high_cond)` 频域重建，t=400 时 w_low=1.1、w_high=1.0，state 逐次存缩放后 delta）；iter5 整批 compute（delta 重算覆盖）。`uncond_skipped` 序列 `[F,T,T,T,T,F]`。

数据规模：张量 `(2,1,2,8,8)`–`(4,1,2,8,8)` float32，N ≤ 8，alpha ∈ [0, 2]。
