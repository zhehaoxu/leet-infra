# 详解：SD3 的 σ 损失加权与置信度感知交叉熵

## 出处

HuggingFace diffusers 的 `src/diffusers/training_utils.py`：

- `compute_loss_weighting_for_sd3(weighting_scheme, sigmas)`——SD3 训练脚本按 σ 给每个样本的 flow-matching 误差配权重的名册。SD3 论文见 arXiv:2403.03206。
- `compute_confidence_aware_loss(logits, labels, *, lambda_conf, temperature, per_token_weights, ignore_index)`——token 级分类头上「纯 CE + 仅在已命中 token 上压熵」的组合损失。

**回源修正**：立项清单里曾把名册写成「sigma_sqrt / cosmap / linear / quadratic」四型，但当前上游（`main` 分支，逐行核对）只有三支：`sigma_sqrt`、`cosmap`，以及兜底的 `ones_like`，并不存在 `linear` / `quadratic` 分支。本题只出上游真实存在的三支，不臆造名册项。

## 核心与易错点

1. **`sigma_sqrt` 里那句 `.float()`**：上游写的是 `(sigmas**-2.0).float()`。float64 的 σ 进来、float32 的权重出去，而 `cosmap` 与 `ones_like` 都不过 `.float()`，dtype 跟着输入走。这是本条最容易被「顺手统一 dtype」改掉的语义。
2. **名册匹配是严格字符串相等，且没有 else 报错**：`"none"`、空串、`"cos_map"`、`"SIGMA_SQRT"` 全都静默落到 `ones_like`。分布式训练脚本里靠这个兜底，改成抛异常会破坏上游调用方。
3. **`sigma_sqrt` 在 float32 下的数值**：`0.1**-2.0` 的 float32 结果是 `99.999992` 而不是 `100`。示例里保留这个值，是因为它正好说明「权重链上没有 float64 中间量」。
4. **`cosmap` 的形状**：`bot = 1 - 2σ + 2σ² = 2(σ-1/2)² + 1/2`，关于 σ=1/2 对称，中点取最小值 1/2（权重最大值 \(4/\pi\)），两端 σ→0、σ→1 都回到 \(2/\pi\)。
5. **`lambda_conf == 0` 的短路**：上游这时不算熵，直接返回 `(loss_sft, loss_sft, 0)`。第一个返回值与第二个**完全相等**，第三个恰好是 0；实现若还老老实实算一遍熵再加 0，数值上可能因为浮点路径不同而对不上。
6. **温度只进熵**：CE 用原始 logits，温度只作用在 `softmax(logits.float()/temperature)` 上。把温度也塞进 CE 会同时改变 `loss_sft`，这是最常见的错法。
7. **`logits.float()` 的降精度位置**：熵一律在 float32 上算完再 `.to(logits.dtype)` 回填。float64 logits 的 `loss_conf` 因此只有约 1e-7 的相对精度，oracle 对拍要给到 `1e-6`。
8. **两个分母的 `clamp_min(1)`**：`loss_sft` 的分母是 `Σ weights·valid`，`loss_conf` 的分母是 `Σ weights·correct`。权重全为 0.25、有效 token 有两个时，分母是 0.5 → 被截到 1，损失变成「0.25 倍的两项之和」而不是平均值。权重为 0 的位置既不进分子也不进分母。
9. **`ignore_index` 可以是合法类号**：它是普通整数比较，不要求是负数。`ignore_index=0` 且标签为 0 时，该位置即便 argmax 命中也被整类排除——`correct` 里的 `valid &` 就是这个作用。
10. **`argmax` 的并列语义**：`torch.argmax` 取第一个最大值下标，oracle 里手写 `_argmax` 必须用严格大于才一致。

## 对拍口径

独立 `_oracle` 不调用任何损失算子：逐 token 用 `math.log`/`math.exp` 的 max-trick 算 logsumexp 得 CE，用 python 标量做 softmax 得熵，两个分母与 `max(..., 1.0)` 截断也手写。九组断言：

- 示例三档权重与示例损失（`loss`/`loss_sft`/`loss_conf` 逐位到 1e-6）＋ `lambda_conf=0` 的短路；
- dtype 矩阵：float64/float32 × 三支名册，`sigma_sqrt` 必为 float32、`cosmap` 跟随输入；
- 名册兜底：五种「非精确匹配」字符串都返回 ones 且不报错；
- `cosmap` 的对称性、中点最大值 \(4/\pi\)、端点 \(2/\pi\)，以及 19 点栅格对拍；
- `ignore_index` 排除（含 `ignore_index=0` 的合法类号）与命中正对照；
- 置信度项只统计命中位置（未命中位置不进、全未命中时退化为 0）；
- 温度只改熵：四档温度下 `loss_sft` 用 `torch.equal` 判完全一致，`loss_conf` 严格单调递增；
- 分母截断：权重 0.25×2 → 0.25·Σnll，权重置 0 的位置被排除，权重 dtype 跟随 logits；
- 两组 (V,L,B) × 3 档 λ × 3 档温度 × {无权重, 随机权重} 的逐位对拍，以及 22 条非法输入。

## 复杂度

权重部分是 \(O(|\sigma|)\) 的逐元素运算；损失部分是 \(O(\prod\!(\cdots, V))\) 的一次 CE + 一次 softmax + 两次归约，无额外缓冲。
