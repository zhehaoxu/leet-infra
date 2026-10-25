# 详解：GRPO 组内标准化优势

## 出处
Shao et al., *DeepSeekMath*（arXiv:2402.03300）提出 GRPO：对同一 prompt 的一组 \(G\) 个输出，用组内 reward 的均值与标准差归一化得到优势 \(\hat A_i=(r_i-\text{mean})/\text{std}\)，替代 PPO 的 critic。本题只实现这一步「reward → advantage」。

## 核心与易错点
1. **按组归一化、非全局**：reshape 成 \((N/G, G)\) 后沿组内(dim=1)求 mean/std，各 prompt 一组，互不干扰。全局归一化会错。
2. **总体标准差 ddof=0**：`std` 除以 \(G\)（`unbiased=False`），与多数 GRPO 实现一致；用 \(G-1\) 会得不同值。
3. **连续分组**：扁平 rewards 前 \(G\) 个是组 0，接下来 \(G\) 个是组 1；不是交错/随机。
4. **零方差退化**：组内 reward 全相等时 std=0（GRPO 常加极小 ε 防除零）；本题给定示例 std>0，判题用严格 \((r-\text{mean})/\text{std}\)、测试也保证非零方差组。

## 对拍口径
参考解 `view(-1,G)` 后向量化 `(x-mean)/std(unbiased=False)`。判题 `_oracle` 逐组用纯 Python 求均值、总体方差开方、逐元素相减除（结构不同、数值等价），`allclose(atol=1e-5,rtol=1e-5)`。另设「每组均值归零」「每组样本方差≈1」「跨组独立」断言。

## 复杂度
\(O(N)\)。
