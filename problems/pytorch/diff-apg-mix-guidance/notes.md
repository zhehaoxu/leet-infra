# 详解：APG-Mix 的 CFG 预热与正交投影引导

## 出处

*Adaptive Projected Guidance*（arXiv:2410.02416）。本题口径对齐 HuggingFace diffusers 的 `src/diffusers/guiders/adaptive_projected_guidance_mix.py`（`forward`、`normalized_guidance`、`MomentumBuffer`），该类用于 HunyuanImage-2.1 的采样管线。原版 APG 是 `guiders/adaptive_projected_guidance.py`，两者共享投影算子，区别只在「是否先用 CFG 预热动量、再切投影」。APG 本体已由 `diff-apg-guidance`（order 501）覆盖平行/正交分解与范数截断，本题只考**两部分拼装 + 跨步动量账**。

上游有一处命名与我们不同：guider 的配置项 `adaptive_projected_guidance_rescale`（默认 10.0）并没有做任何 rescale，而是作为 `norm_threshold` 位置参数传进 `normalized_guidance`。本题直接把它叫 `norm_threshold`，避免歧义。

## 核心与易错点

1. **CFG 分支用未平滑的差值**：源码里 CFG 分支先 `update_momentum_buffer(...)`，然后仍然用 `shift = pred_cond - pred_uncond` 做平移。顺手把 `shift` 换成 `running` 是最常见的错，它会让预热阶段的结果依赖历史动量——但此时 APG 还没开启，输出应当与普通 CFG 逐位相同。
2. **缓冲区保存截断前的值**：`normalized_guidance` 先更新 buffer，再把 `diff` 拷出来做范数截断。若把截断后的值写回 buffer，后续每一步的动量都会被反复压缩，误差随时间累积（本题的 10 步链式对拍专门抓这个）。
3. **APG 起点是严格大于**：`step > adaptive_projected_guidance_start_step`。写成 `>=` 会让 `apg_start_step=5` 时第 5 步就切到投影引导，比预期早一步。
4. **APG 以 CFG 为前提**：`_is_apg_enabled` 第一行就 `if not self._is_cfg_enabled(): return False`。所以 `cfg_scale` 退化（默认口径下 `isclose(cfg, 1.0)`）时，即使 `apg_scale` 正常也返回 `pred_cond`，且**不更新动量**。
5. **范数在除 batch 外的所有维度上求**：源码 `dim = [-i for i in range(1, len(diff.shape))]`，对 `(B,C,H,W)` 就是 `(C,H,W)` 整体求范数，不是逐像素。写成 `dim=-1` 会让每个像素各自按阈值截断，投影方向完全变样。
6. **负动量**：`momentum` 默认 `-0.5`，`running = d + \mu\cdot running\_prev` 因此是在当前差值与上一拍差值之间做**外推**（`0.5d`）。把它当指数滑动平均写成 `(1-\mu)d + \mu\,prev` 会得到另一个数。
7. **窗口外不动动量**：`not cfg_on` 分支直接返回 `pred_cond`，动量不更新；若 `buffer` 传的是 `None`，返回值也应是 `None`（还没建立状态），不要凭空造一个零张量。
8. **`use_original_formulation` 换的是两件东西**：基准从 `pred_uncond` 换成 `pred_cond`，同时「尺度退化」的判据从 `isclose(s, 1.0)` 换成 `isclose(s, 0.0)`。只改前者会在 `cfg_scale=0.0` 时错误地继续引导。

## 对拍口径

参考解全程 float64 计算、float32 返回，动量状态按 float32 携带（与真实管线一致，会带来约 \(10^{-7}\) 的舍入差异）。判题 `_oracle` 不复用任何张量运算：窗口用 python `int()` 与 `math.isclose` 重判，动量递推、范数、内积、正交分解全部用逐元素 python 标量循环（先 reshape 成 `(batch, width)`）。八组断言：给定示例三行、预热阶段「step 0–5 与 CFG 逐位相同且动量逐拍递推」、三种动量配置（`None` / `0.0` / 负值）与输入不被就地修改、阈值截断（关闭、触发、只缩放不改方向）、投影性质（\(\eta=1\) 退化为 CFG、\(\eta=0\) 与 \(\epsilon_c\) 正交、模长不增）、窗口与 `use_original_formulation` 的开关语义、四组参数各 10 步的链式对拍（含窗口非全区间）、以及非法输入。阈值 `atol=1e-5`。

## 复杂度

单步逐元素 \(O(\text{numel})\)，外加两次按样本的归约（范数、内积）；动量状态额外占用一份与预测同形的缓冲，跨步常驻。
