# 详解：PAG 的双尺度引导与两套窗口

## 出处

*Perturbed Attention Guidance*（arXiv:2403.17377）。实现口径对齐 HuggingFace diffusers 的 `src/diffusers/guiders/perturbed_attention_guidance.py`：`forward`（L213–240）给出四个分支的完整组合，`_is_cfg_enabled`（L258–274）与 `_is_slg_enabled`（L277–289）给出两套窗口的判定——前者是 `<= step <`，后者是 `step` 严格居中。该类复用了 `SkipLayerGuidance` 的参数命名，所以源码里 PAG 的尺度字段叫 `skip_layer_guidance_scale`、窗口字段叫 `skip_layer_guidance_start/stop`。

## 核心与易错点

1. **两套窗口的端点语义不同**：CFG 是 \([\,lo, hi)\)，PAG 是 \((lo, hi)\)。把 PAG 也写成左闭会在 `step = int(perturbed_start*N)` 那一步多做一次扰动引导，默认参数下就是 `step=0` 被错误地拉进 PAG 分支。
2. **整数取整是截断**：`int(...)` 对非负数等价于 `floor`，但写成 `round(...)` 或 `ceil(...)` 都会让窗口边界整体偏移。步数很少时（例如 \(N=3\)、`perturbed_stop=0.2`）PAG 窗口会退化成空区间，此时该分支永不生效。
3. **"关闭"有两种来源**：窗口外，或者尺度退化（\(s\approx 1\) 关 CFG、\(s_{\text{skip}}\approx 0\) 关 PAG）。两者是同一个 `and not isclose(...)` 表达式，不要按两种独立情况分开判断，否则会出现「窗口外但尺度也退化」时的分支错位。
4. **PAG 分支的基准是扰动预测**：只用 PAG 时输出为 \(\epsilon_{\text{skip}}+s_{\text{skip}}(\epsilon_c-\epsilon_{\text{skip}})\)，不是以 \(\epsilon_u\) 为基准。写成 \(\epsilon_u + \dots\) 会在「只有 PAG 的窗口」里给出完全错误的结果。
5. **双开时两个扰动项直接相加，共用同一个 \(\epsilon_u\) 基准**：\(\epsilon_u+s\Delta_{\text{cfg}}+s_{\text{skip}}\Delta_{\text{pag}}\)。把 PAG 项改成在前者的结果上再叠加（即基准变成已引导的预测）是另一个常见错误，会让可加性断言失败。
6. **缺省分支允许 `None`**：diffusers 里没开启的条件根本不会被推理出来，所以 `pred_uncond` / `pred_cond_skip` 都是可选参数。判题既检查「关闭时可省」，也检查「生效却缺省必须报错」。

## 对拍口径

参考解先在 float64 下算好窗口与分支，逐元素组合后转 float32。判题 `_oracle` 不复用任何张量运算：窗口用 python 的 `int()` 与 `math.isclose` 重判，组合用逐元素 python 标量循环（先 reshape 成 `(batch, width)` 再展平）。结构断言五组：窗口边界的四个特征步（左端、右端、端内、端外）、尺度退化（\(s=1\) / \(s_{\text{skip}}=0\) / 两者）、`both - cfg == s_skip·Δ_pag` 与 `both - pag == s·Δ_cfg` 的可加性、可选分支的 `None` 语义、以及非法输入。阈值 `allclose(atol=1e-5)`。

## 复杂度

逐元素 \(O(N)\)，无归约、无额外缓冲；四段预测各读一遍。
