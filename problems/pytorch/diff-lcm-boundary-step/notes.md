# 详解：LCM 边界条件系数与少步去噪单步

## 出处

HuggingFace diffusers 的 `src/diffusers/schedulers/scheduling_lcm.py`：`get_scalings_for_boundary_condition_discrete`（\(c_{\text{skip}}\) / \(c_{\text{out}}\)，\(\sigma_d\) 写死 0.5）、`step`（\(x_0\) 换算、clip/threshold、边界条件混合、按 \(\bar\alpha_{\text{prev}}\) 重加噪）、`set_timesteps`（LCM 蒸馏栅格 `arange(1, original_steps+1)*k - 1`，再按 `floor(linspace(0, N, endpoint=False))` 子采样）。算法来自 Luo et al., 2310.04378（Latent Consistency Models）与其官方实现的 `get_scalings_for_boundary_condition_discrete`。

## 核心与易错点

1. **\(c_{\text{out}}\) 少了 \(\sigma_d\)**：LCM 用 \(s/\sqrt{s^2+\sigma_d^2}\)，EDM 用 \(s\sigma_d/\sqrt{s^2+\sigma_d^2}\)。\(\sigma_d=0.5\) 时两者恰好差 2 倍，而且 LCM 蒸馏出来的权重就是按这个系数训练的——**不要按 EDM 公式"修正"**，否则推理结果整体差一倍。连带后果：EDM 里成立的 \(c_{\text{skip}}^2+c_{\text{out}}^2=c_{\text{skip}}\) 在 LCM 下不成立。
2. **系数由 timestep 下标决定，不是由 \(\sigma\) 决定**：\(s=t\cdot\text{timestep\_scaling}\)，`timestep_scaling` 默认 10。LCM 的 timestep 是训练栅格的下标（0..999），所以 \(s\) 在 0..9990 之间；把它当成连续时间 \(t\in[0,1]\) 会得到完全不同的系数。
3. **\(\sigma_d\) 是在函数体里赋值的**：上游 `self.sigma_data = 0.5` 写在 `get_scalings_for_boundary_condition_discrete` 内部，每次调用都覆盖一次，且不可配置。本题照抄这一常量。
4. **\(t=0\) 时退化**：\(c_{\text{skip}}=1\)、\(c_{\text{out}}=0\)，于是 `denoised` 精确等于输入样本、与模型输出无关。真实的 LCM 栅格最小值是 \(k-1\)（\(k=T//\text{original\_steps}\)，如 1000/50 → 19），并不取 0，所以实际推理不会撞上这个退化点；本题允许调度里出现 0，并把它作为一条显式断言。
5. **`prev_timestep` 取数组的下一项，不是 \(t-1\)**：`timesteps[step_index+1]`。因为 LCM 栅格是等距子采样（间隔 \(k\)），\(t-1\) 会落到别的噪声级上。最后一步没有下一项时沿用当前 timestep——该值不参与计算（直接返回 `denoised`）。
6. **最后一步不重加噪**：判据是 `step_index != num_inference_steps - 1`，用的是**配置的步数**而不是"当前 timestep 是否为 0"。因此单步采样（长度 1 的调度）中唯一的一步就返回纯 `denoised`，这正是 LCM 1-step 推理的形态。
7. **\(\bar\alpha_{\text{prev}}\) 的下标越界保护**：`alphas_cumprod[prev_timestep] if prev_timestep >= 0 else final_alpha_cumprod`。本题把 `final_alpha_cumprod` 取 1.0（上游 `set_alpha_to_one=True` 的默认），所以负 timestep 会得到 \(\sqrt{1}=1\)、\(\sqrt{0}=0\)，等价于直接把 `denoised` 当结果。
8. **\(x_0\) 换算用的是正确的 \(\alpha_t\) 形式**：`v_prediction` 支是 \(\sqrt{\bar\alpha_t}x-\sqrt{1-\bar\alpha_t}v\)，与 `diff-euler-ancestral-step` 里那个少乘 \(\alpha\) 的写法不同——同一个仓库里两处 v-prediction 换算并不一致，值得对照着读。
9. **裁剪有两层**：`thresholding=True` 走动态阈值，否则 `clip_sample=True` 走固定范围裁剪（默认 \([-1,1]\)）。动态阈值已在 `diff-dpm-x0-conversion-threshold` 单独出题，本题只保留 `clip_sample`，并要求它作用在 \(x_0\) 上、**混合之前**。
10. **数值口径**：上游按 `model_output.dtype` 计算并返回；本题统一 float64 内部计算、返回 float32。

## 对拍口径

判题用 numpy 的 `_oracle` 按同样公式重做系数、\(x_0\)、裁剪、混合与重加噪（输入先压到 float32，与参考解同一批舍入值）：

- **示例值**：`t=9` / `t=2` 的 `c_skip`、`c_out`、`denoised`、`prev` 六位小数硬编码，最后一步断言 `prev` 与 `denoised` 逐位相同；
- **系数表**：闭式复算 \(c_{\text{skip}}\) / \(c_{\text{out}}\)、断言 \(c_{\text{out}}=2\times\)EDM 版本、断言 EDM 的 \(c_{\text{skip}}^2+c_{\text{out}}^2=c_{\text{skip}}\) 不成立、四个 timestep 上单调性（\(c_{\text{skip}}\) 降、\(c_{\text{out}}\) 升）、\(t=0\) 的退化值与 `denoised==sample`、`timestep_scaling=5` 与 10 的关系；
- **预测类型**：由同一 \(x_0\) 反解 \(\epsilon\)、\(v\)，三条分支给出相同 `denoised`；`sample` 支等于 \(c_{\text{out}}\cdot\text{model}+c_{\text{skip}}\cdot x\)；
- **裁剪**：开/关不同、`clip_sample_range=0.4` 更紧、裁剪发生在混合之前；
- **重加噪**：零噪声时 `prev=√ᾱ_prev·denoised`、单位噪声时差值恰为 \(\sqrt{1-\bar\alpha_{\text{prev}}}\)、最后一步换任意噪声结果不变、单步调度即最后一步；
- **`prev_timestep` 来源**：同一当前 timestep 配不同长度的调度，`prev` 不同而 `denoised` 相同；重复 timestep 时取数组下一项而非 \(t-1\)；
- **组合对拍**：2 种形状 × 4 组调度（含单步、重复值、含 0）× 每步 × 3 种预测类型，外加 `clip_sample=False` 与 `clip_sample_range=0.3` 两轮；
- **非法输入**：21 组（非张量 / 整型 / 形状不符 / `alphas_cumprod` 为 list / 二维 / 含 1.0 / 含 0 / `timesteps` 为 list / 浮点 / 越界 / 含负 / 二维、`step_index` 越界 / `bool`、未知预测类型、`clip_sample=1`、`clip_sample_range=0`、`timestep_scaling` 为负、非最后一步缺 `noise`、`noise` 形状不符）都必须抛 `ValueError`。

共八组断言。

## 复杂度

每个元素常数次乘加，\(O(B\cdot C\cdot H\cdot W)\) 时间与空间（`denoised` 与 `prev_sample` 各一份）；系数是标量，与张量规模无关。
