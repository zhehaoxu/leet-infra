# 详解：VE/EDM 预条件四系数与带 churn 的 Euler 步

## 出处

HuggingFace diffusers 的 `src/diffusers/schedulers/scheduling_edm_euler.py`：`precondition_inputs`（`c_in`）、`precondition_noise`（`c_noise = 0.25·ln σ`）、`precondition_outputs`（`c_skip` / `c_out` 与合成）、`step`（churn + Euler 步）。公式本身来自 Karras et al. 2206.00364 与 NVlabs/edm 的 `EDMPrecond`。`set_timesteps` 里 `self.timesteps = self.precondition_noise(sigmas)` 说明 EDM 一族把「时间条件」直接定义为 \(0.25\ln\sigma\)，与 VP 侧的 \(\bar\alpha\) 口径不同。

## 核心与易错点

1. **四个系数分属两个 σ**：`c_in` / `c_noise` 用调度器当前的 \(\sigma\)，而 `c_skip` / `c_out` 用 churn 之后的 \(\hat\sigma\)。上游之所以如此，是因为 `scale_model_input` 在 `step` 之前被 pipeline 调用（那时还没加噪），`precondition_outputs` 在 `step` 内部才用 \(\hat\sigma\)。把两侧都写成 \(\hat\sigma\) 是最隐蔽的错——不加 churn 时两者相同，所有测试都会通过，一开 churn 就全错。
2. **`c_out` 的符号是预测类型**：\(\epsilon\) 支取 \(+\hat\sigma\sigma_d/\sqrt{\hat\sigma^2+\sigma_d^2}\)，`v_prediction` 支取负号。EDM 论文里 \(F\) 预测的是「带负号的缩放速度」，所以换成 v 预测只需翻转符号；测试用恒等式 \(D_v = 2c_{\text{skip}}x - D_\epsilon\) 验证。
3. **\(\gamma\) 的分母是 `len(sigmas) - 1`**：`sigmas` 末尾已经追加了终端 0，所以步数 \(K = \text{len}-1\)。写成 `len(sigmas)` 会让 \(\gamma\) 上限对应的 `s_churn` 偏小。
4. **窗口判定用未加噪的 \(\sigma\)**：`s_tmin <= sigma <= s_tmax`，不是 \(\hat\sigma\)。窗口外 \(\gamma=0\)，此时 `noise` 完全不参与计算——实现里若先无条件加噪再乘 0，会因为 `sqrt(0)` 与 NaN 噪声引入 0·NaN 的坑。
5. **`dt` 用的是 \(\hat\sigma\)**：`dt = sigmas[i+1] - sigma_hat`。若误用 \(\sigma\)，步长会偏大，而且最后一步不再有「精确落到 `denoised`」的性质。
6. **最后一步的恒等式**：`sigmas[-1] == 0` 时 \(dt = -\hat\sigma\)，于是 `prev = x + (x - D)/\hat\sigma·(-\hat\sigma) = D`。这是本题最有用的自检：任何一步只要 \(\hat\sigma = \text{sigmas}[i+1]\) 就会退化成直接跳到来噪样本。
7. **\(\sigma\to t\) 是单调对数刻度**：`c_noise = 0.25 ln σ`，反解 \(\sigma = e^{4t}\)。它与 `diff-v-prediction-convert` 的 VP 侧 \((\bar\alpha,\sigma)\) 换算是两套口径，互不覆盖。
8. **数值精度**：上游把 `sample` 升到 float32（避免 MPS 上 float16 计算误差），本题统一在 float64 里算再降回 float32，因此示例数值与 float64 手算完全一致。
9. **未纳入的部分**：`preconditioning_threshold` 属于训练侧的 `KarrasDenoiser`（SD3 的 x₀→v 阈值转换），不在这份调度器源码里；为保持「只出一道源内算法」的纪律，本题不臆造该分支。

## 对拍口径

判题用纯 Python 标量 oracle（`_oracle`）以嵌套 list 逐元素重做加噪、四系数与 Euler 步，再与参考解比对：

- **示例值**：三步 + churn 的五组输出（`model_input` / `c_noise` / `denoised` / `prev` / `sigmas_hat`）全部六位小数硬编码，并断言最后一步 `prev` 与 `denoised` 逐位相同；
- **输入侧与 churn 解耦**：`s_churn=50` 时 `model_input` 仍等于「原始样本 × c_in」，而不是「加噪样本 × c_in」；\(\gamma=0\) 时传入 `noise` 与不传结果逐位一致；
- **窗口与截断**：`s_tmin=5`、`s_tmax=3`、`s_churn=0` 三种配置下 \(\hat\sigma=\sigma\)；窗口内 `s_churn=0.6` 给 \(\hat\sigma = 4·(1+0.2)\)，`s_churn=9` 被截断为 \(4\sqrt2\)；
- **最后一步**：三组不同长度的 [σ 表] 下 `prev_sample == denoised`；
- **预测类型**：\(D_v = 2c_{\text{skip}}x - D_\epsilon\) 在每步成立；
- **时间刻度**：`c_noise` 与 \(0.25\ln\sigma\) 一致，且 `exp(4·c_noise)` 还原 \(\sigma\)；
- **组合对拍**：3 组 σ 表 × 每步 × 5 种配置（含 `v_prediction`、`s_noise=0.7`、`sigma_data=0.2` 与窗口内 churn）与 oracle 逐项一致；
- **非法输入**：25 组（非张量/整型/形状不符/空张量、sigmas 为 list/二维/整型/长度为 1/含负/递增/含 NaN、`step_index` 越界或非整数、`sigma_data=0`/负数、`s_churn`/`s_noise` 为负、`s_tmin>s_tmax`、未知预测类型、churn 缺 noise、noise 形状或 dtype 不符、当前 σ 为 0）都必须抛 `ValueError`。

共八组断言。

## 复杂度

对样本张量的每个元素做常数次乘加，\(O(B\cdot H\cdot W)\) 时间、\(O(B\cdot H\cdot W)\) 空间（只额外生成模型输入、去噪输出与下一步样本各一份）。
