# 详解：DPM-Solver 口径转换与动态阈值

## 出处

HuggingFace diffusers 的 `src/diffusers/schedulers/scheduling_dpmsolver_multistep.py`：`convert_model_output`（按 `algorithm_type` 分成数据口径 / 噪声口径两大支，各自再按 `prediction_type` 分四 / 三支）、`_threshold_sample`（标注为 copied from `scheduling_ddpm.DDPMScheduler._threshold_sample`）、`_sigma_to_alpha_sigma_t`（`alpha = 1/√(σ²+1)`、`sigma_t = σ·alpha`，或 `use_flow_sigmas` 下的 `alpha = 1-σ`、`sigma_t = σ`）。动态阈值公式来自 Imagen（2205.11487）。

## 核心与易错点

1. **两种口径的输出完全不同**：`dpmsolver++` 返回 \(x_0\)，`dpmsolver` 返回 \(\epsilon\)。同一个 `prediction_type="epsilon"` 在两种口径下的分支不是「同一个公式换个写法」——数据口径要把 \(\epsilon\) 换算成 \(x_0\)，噪声口径直接原样返回。答案的形状虽然一样，数值含义相反。
2. **`sample` 预测是恒等分支**：数据口径下 `x0_pred = model_output`；噪声口径下要先 `eps = (x - α·model_output)/σ_t`。这一支最容易被误写成「也返回 model_output」。
3. **`flow_prediction` 只存在于数据口径**：它用原始 \(\sigma\)（`x0 = sample - σ·model_output`）而不是 \(\alpha/\sigma_t\)，且此时 `model_output` 是速度 \(v=\epsilon-x_0\)。噪声口径的 `else` 会显式抛 `ValueError`。把 flow 支写成 `alpha/sigma_t` 在 `use_flow_sigmas=True` 时恰好数值相同（因为 \(\alpha=1-\sigma\) 但 \(\sigma_t=\sigma\ne\sigma\alpha\)）……所以务必照抄源码：flow 支根本不做 \(\alpha\) 换算。
4. **噪声口径的阈值化要往返两次**：先 `x0 = (x-σ_t·ε)/α`，再 `x0 = threshold(x0)`，最后 `ε = (x-α·x0)/σ_t`。只对 \(\epsilon\) 直接阈值是错的——阈值作用在 \(x_0\) 上，\(\epsilon\) 只是中间的运输工具。
5. **`learned` 方差只影响 epsilon 分支**：`model_output[:, :3]` 取的是前三个通道（DDPM 的 mean 通道），后面的通道是方差，DPM-Solver 只要 mean。因此 `model_output` 是 \((B,6,\dots)\) 而 `sample` 是 \((B,3,\dots)\)——形状校验必须先切开再比。
6. **分位数是逐样本算的**：`torch.quantile(abs_sample, ratio, dim=1)` 在展平后的 \((B, C\cdot H\cdot W)\) 上沿 dim=1 取，所以每个样本有自己的 \(s\)，batch 之间互不影响。误用全局分位数会让整批共享一个尺度。
7. **`s` 的双侧钳位**：`clamp(s, min=1, max=sample_max_value)`。`min=1` 让 \(|x_0|\le 1\) 的样本原样通过（等价于普通裁剪），`max` 则防止分位数被极端值拉爆。因此输出**永远**落在 \([-1,1]\) 内，即使 `sample_max_value` 很大。
8. **`sample_max_value` 小于 1 无意义**：`min=1` 与 `max=sample_max_value<1` 冲突，`torch.clamp` 在这种情况下行为不明确，所以本题要求它 \(\ge 1\)。
9. **`use_flow_sigmas` 下 \(\sigma\ge1\) 会除零**：\(\alpha=1-\sigma\)，\(\sigma=1\) 时数据口径的 `epsilon` 支除以 0。本题要求 \(0<\sigma<1\)。
10. **数值口径**：上游为了 quantile 会把 half 升 float32 再降回；本题统一在 float64 里算、返回 float32，因此示例数值与 float64 手算完全一致。

## 对拍口径

判题用 numpy 实现独立 oracle（`_oracle` + `_threshold_oracle`，分位数用 `np.quantile(method="linear")`），按同样的分支重做整条换算，再与参考解比对：

- **示例值**：三组六位小数硬编码（数据口径 \(\epsilon\) / \(v\)、噪声口径 \(v\)、阈值化），外加「噪声口径 \(\epsilon\) 原样返回」的逐位相等；
- **预测类型恒等式**：由 \(x=\alpha x_0+\sigma_t\epsilon\) 出发，反解 \(\epsilon\)、\(v=\alpha\epsilon-\sigma_t x_0\)，验证数据口径能还原 \(x_0\)；`sample` 支逐位返回；flow 支用 \(v=\epsilon-x_0\) 验证还原；
- **噪声口径互逆**：`dpmsolver` 的 `sample` / `v_prediction` 支还原 \(\epsilon\)，且噪声口径 \(\epsilon\) 与数据口径 \(\epsilon\) 互逆；
- **阈值化数学**：\(|x_0|\le1\) 的输入逐位不变、输出恒在 \([-1,1]\)、`ratio=1.0` 时等于除以逐样本最大绝对值、改变一个样本不影响另一个样本、`sample_max_value` 在紧/松两种取值下给出不同结果；
- **flow 口径**：\(\alpha=1-\sigma\) 下能还原 \(x_0\)，且与 VP 口径结果明显不同（防止两支被写成同一段代码）；
- **learned 方差**：6 通道输入与直接传 3 通道逐位相同（两条口径各验一次），并叠加阈值化；
- **组合对拍**：2 种形状 × 3 个 \(\sigma\) × 14 组配置（覆盖四种 `prediction_type`、两种口径、阈值化、`sample_max_value`/`ratio` 变化、flow 组合），再加 2 组 learned 配置；
- **非法输入**：21 组（非张量 / 整型 / 形状不符 / 空张量 / 1 维、\(\sigma\) 为 0 / 负 / NaN / `bool`、未知 `prediction_type` / `algorithm_type` / `variance_type`、`thresholding=1`、`use_flow_sigmas="yes"`、`ratio` 为 0 或 >1、`sample_max_value<1`、flow 下 \(\sigma\ge1\)、噪声口径配 flow、learned 配非 epsilon、learned 形状不符）都必须抛 `ValueError`。

共八组断言。

## 复杂度

每个元素常数次乘加，加上一次逐样本的排序取分位数：\(O(B\cdot N\log N)\) 时间（\(N=C\cdot H\cdot W\)）、\(O(B\cdot N)\) 空间。
