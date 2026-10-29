# 详解：LTX-2 三路增量引导合成

## 出处

LTX-2 的 guider 实现对齐 HuggingFace diffusers 的 `src/diffusers/guiders/ltx2_guidance.py`：`forward` 里的注释直接写明了增量形式 `x0_cond + (gs-1)*(cond-uncond) + stg*(cond-stg) + (mod-1)*(cond-modality)`，三个分量各由 `is_cfg_enabled` / `is_stg_enabled` / `is_modality_enabled` 判定，且共享 `_within_window` 的半开区间。STG 的论文出处是 *Spatio-Temporal Guidance for Video Diffusion*（arXiv:2411.18664）；STG 单独的「跳层」分支合成已由 `diff-pag-dual-guidance`（order 504）覆盖，本题不重复那套四分支骨架，只出 LTX-2 特有的三路增量形式。

## 核心与易错点

1. **全部以条件预测为基准**：CFG 的原生写法是 `uncond + s*(cond-uncond)`，改写为增量形式是 `cond + (s-1)*(cond-uncond)`，二者在 \(s\) 处代数等价——但**只在单路时等价**。三路叠加时若各自带自己的基准（`uncond` / `stg` / `modality`），结果会偏离源头；源码的写法是统一从 \(\epsilon_c\) 出发，所以三项可加。
2. **恒等点不同**：`cfg_scale` 与 `modality_scale` 的恒等点是 1.0，`stg_scale` 的恒等点是 0.0。这三个默认值（1.0 / 0.0 / 1.0）就是「该项关闭」，源码用 `math.isclose` 判定，因此 `1.0 + 1e-12` 也算关闭。
3. **开关是 `and not isclose`，不是「把差值乘零」**：两者的区别在于 \(s_c=1\) 时是否还要跑无条件分支。判题里对「关闭时允许传 `None`」和「生效时缺 `None` 必须报错」双向断言，正是为了卡住「先算后乘零」的实现。
4. **窗口只有一套**：与 PAG/SLG 那种「CFG 一套、扰动一套」不同，LTX-2 的两个 guider（video / audio）各自持有尺度，但窗口对三路统一，且是严格左闭右开。
5. **`stg_scale` 可以为负**：STG 的反向引导在实践中用来抑制过强的时域一致性，因此本题不把 STG 尺度限制成非负，而 CFG/模态尺度也允许任何有限实数（不做下界限制）。
6. **多模态 guider 的实例化不属于本题**：真实管线里 video 与 audio 各有一个 guider 实例、各自持有尺度，且 `spatio_temporal_guidance_blocks` 只读 video guider。那是装配细节，本题只取「单模态单步合成」。

## 对拍口径

参考解 float64 计算、float32 返回。判题 `_oracle` 用 python 标量重做窗口与三项累加（`reshape(B,-1)` 后逐元素），七组断言：示例三行（含默认退化）、恒等点退化（可完全缺省三路预测）、三项可加性（`full - cond == Σ 增量`，且单项结果都从 `cond` 出发）、尺度语义（cfg/modality 的 2.0 等于加一份差值，stg 的 1.0 等于加一份差值；`isclose` 退化)、窗口边界（左闭右开、窗口外无需分支预测）、五组参数各 10 步的全量对拍、非法输入。阈值 `atol=1e-5`。

## 复杂度

逐元素 \(O(\text{numel})\)，四段预测各读一遍，无归约。
