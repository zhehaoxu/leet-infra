# 量化：PB-LLM 部分二值化——算法详解

## 1. 低比特量化的盲点

把所有权重压到 1～2 bit 时，误差不再是均匀分布的：LLM 权重里少数大幅度值（显著权重）对输出贡献很大，把它们和大量小权重一起二值化，会让模型能力断崖式下降。PB-LLM（Partially Binarized LLMs, 2023, arXiv:2310.00034）的观察是：小幅度权重更"冗余"，可以承受二值化；把低比特预算集中给它们，大权重保留较高位宽，就能在同等平均位宽下显著降低困惑度。

上游流程分三步：按幅度选出 `low_frac` 比例的权重作为 low 组；low 组用 xnor（1-bit + 均值/尺度）表示；其余用 `high_bit` 位非对称量化；最后按掩码拼合。本题实现其中的 RTN 分支（`disable_gptq=True`），跳过 Hessian 误差补偿。

## 2. 掩码与 low 通路

掩码阈值来自全局排序：`thresh = sorted(|W|.flatten())[int(numel * low_frac)]`，`mask = |W| <= thresh`。用 `<=` 而不是 `<` 意味着并列值会一起入选，实际比例可能略高于 `low_frac`；`low_frac=0` 时只选中最小值（可能不止一个）。

low 通路的标定在 `masked = W * mask` 上进行：

\[
\mu=\operatorname{mean}(W\odot m),\qquad
s=\operatorname{mean}(|W\odot m-\mu|)
\]

注意两个细节：

- 未入选位置以 0 参与 `mean`，分母是全部元素数 `C`（上游 `LowQuantizer.calibrate` 直接对 `W*mask` 求 `mean(-1)`）；
- `s` 是**平均绝对偏差**，不是方差或标准差。

重建为 `sign(w-μ)·s + μ`。可以把它看成"带中心的三值/二值"：`sign` 给出 ±1 或 0，`s` 控制幅度，`μ` 把中心平移回去。由于 masked 后大部分位置是 0，`μ` 和 `s` 都会被拉小，正好匹配"小权重"的量级。

## 3. high 通路

high 通路逐行统计（`perchannel=True`、`sym=False`）：

\[
x_{\min}=\min(\min(row),0),\quad x_{\max}=\max(\max(row),0)
\]

二者同时为 0 时置为 `(-1,+1)`。`maxq = 2^bits - 1` 是全 1 的无符号码，因此：

\[
s_h=\frac{x_{\max}-x_{\min}}{maxq},\qquad z_h=\operatorname{round}\!\left(\frac{-x_{\min}}{s_h}\right)
\]

量化 `q = clamp(round(w/s_h) + z_h, 0, maxq)`，重建 `(q - z_h)·s_h`。与对称量化不同，正负极值都各占满一个端点，零点是"把最小值对齐到码 0"。

## 4. 实现口径与边界

- **单组**：本题固定 `groupsize=-1`，阈值在整张 `W` 上取；low 的 `mean/scale` 仍逐行统计。上游若设置更小的 `groupsize`，阈值会按列分组分别计算，本题不涉及。
- **只做 RTN**：不实现 GPTQ 的列间误差补偿，因此结果是"先掩码、再两路量化、再拼合"的纯 RTN 版本。
- **全等权重**：阈值等于该值，整个矩阵都进 low 通路；`masked` 恒等于原值，`μ` 等于该常量、`s=0`，重建精确等于常量。
- **全零行**：high 通路按 `(-1,+1)` 得到 `s_h=2/maxq`、`z_h=round(maxq/2)`；若该行被 low 覆盖，重建仍为 0。
- **舍入**：`high_zero` 与 `round(w/s_h)` 都用四舍五入到偶数（与 `torch.round` 一致）；内部 float64 保证判题稳定。
- `low_frac ∈ [0,1)`：等于 1 会让排序下标越界（上游同样没有余量），本题直接抛 `ValueError`。

## 5. 与本合集其他题的分工

- `quant-squeezellm-sensitive-kmeans`：对**敏感通道**做非均匀聚类，本题是按幅度分流的二值/高位混合；
- `llm-int8-outlier-split`：把 outlier 通道单独拆出，本题不拆通道，而是逐元素打掩码；
- `quant-awq-clip-scale-search`：对显著通道做 scale 搜索，本题不做激活侧分析；
- `symmetric-quantize` / `asymmetric-quantize`：均匀整数格点，是本题 high 通路的基础；
- 本题：**幅度掩码 + xnor low 通路 + 逐行非对称 high 通路 + 按掩码拼合**。

## 6. 验证方法

- `check_example`：手算 6 元素示例，覆盖全局阈值、low 统计与 high 重建。
- `check_mask_threshold`：多个 `low_frac` 下与独立计算的阈值/掩码一致；全等权重时掩码全 True。
- `check_low_stats`：`low_mean/low_scale` 与 `W*mask` 上的逐行统计一致（未入选位置按 0 计入）。
- `check_high_stats`：不对称行、全正行、全负行、全零行的 `scale/zero`。
- `check_matches_oracle`：30 组随机配置（含单符号行、常量行）与纯 Python oracle 逐项对拍。
- `check_piecewise_selection`：重建值逐元素等于 mask 选中的 low/high 通路结果。
- `check_all_equal_and_zero_rows`：全等行与全零行的退化行为。
- `check_validation`：`high_bits`、`low_frac`、`W` 的非法入参抛 `ValueError`。
