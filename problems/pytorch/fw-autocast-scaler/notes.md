# 详解：autocast 与 GradScaler 训练循环账

## 出处

PyTorch AMP 文档（*torch.amp*）：`torch.amp.autocast(device_type='cpu')` 的 CPU 默认 dtype 是 `bfloat16`，可显式传 `dtype=torch.float16`；matmul/linear 落在 cast list（降到半精度），而参数的 `.grad` 由 cast 边界换回参数 dtype（fp32）。`torch.amp.GradScaler(device='cpu', enabled=True)` 在近版本 CPU 上可用（判题环境 torch 2.14 实测 `is_enabled()==True`；旧版本 CPU 会打 warning 并自动 disable——行为随版本变，题面数值以参考解实跑为准）。状态机语义：`unscale_` 一个 update 周期只许调一次，第二次 `RuntimeError: unscale_() has already been called on this optimizer since the last update()`；`step` 发现 inf 则跳过、`update` 退避 `backoff_factor`，否则 `_growth_tracker+=1`，满 `growth_interval` 次连升 `growth_factor` 倍并清零。

## 核心与易错点

1. **跳步的信号在权重上，不在返回值里**：新版 `scaler.step()` 无论跳过与否都返回 `None`，只有 `torch.equal(before, after)` 或后续 scale 变化能当账本证据。
2. **scale 只影响下溢保护、不影响数学**：`unscale_` 后梯度回到真值，所以权重轨迹与纯 fp32 训练逐位一致（前提：数值二进制精确——题面全用 dyadic 小数）。
3. **升降轨迹是状态机题**：`[16,8,8,16,8,8]` 完全由「注入 inf 的位置 + growth_interval=2」决定，oracle 用十行纯 Python 重放同一定理，不 import GradScaler。
4. **CPU autocast 的 dtype 规则**：`x@w` 输出 fp16、`mean` 保持 fp16（CPU 的 promotion list 与 CUDA 不同），但 `w.grad` 是 fp32——「混合精度」混合的是前向算力与反向存储精度。
5. **次序违规有硬保护**：unscale→unscale 报错之外，真实工程事故（忘 unscale 就 clip、update 前连续 step）同样由这套状态机抛错兜底，本题的探针是其中最常见的一种。

## 对拍口径

solution 走真 autocast/GradScaler；`_oracle` 纯 Python float 重放：每步手算 \(out=1.5w_0-0.5w_1\)、\(loss=(out-1)^2\)、解析梯度 \(\nabla=2(out-1)x\)、按状态机规则更新 scale/tracker——与前向无关的「跳步冻结权重」直接由 `updated` 布尔账核对。另设三条不变量：跳步后 loss 原地重复、t3 翻倍账、double-unscale 必抛。

## 复杂度

6 步 × 2 参数，微秒级；oracle 同规模。
