# 详解：SFT 逐轮损失权重

## 出处
工程实践中的 per-turn / per-sequence loss 归一技巧（与「token-mean vs sequence-mean」、多轮对话按 turn 平衡贡献同源）。核心不变量：让**每轮对总损失的贡献**等于人为设定的基数，而不随该轮 token 数线性膨胀；短轮不被长轮淹没。本函数把该思路实现成逐 token 权重。

## 核心与易错点
1. **轮 = 行内极大 assistant 连续段**：编号在**每行内**独立从 0 起，不是全局累计；跨行必须重置，否则 `turn_weights` 索引错位。
2. **权重与轮长成反比**：同轮每 token \(=c_t/n_t\)，一轮求和 \(=c_t\)。若忘记除以 \(n_t\)（写成 \(c_t\) 每 token），长轮又会淹没——正好违背本题目的。
3. **`turn_weights` 越界回退 1.0**：某轮 assistant 数超过 `len(turn_weights)` 时，多出来的轮按基数 1.0。
4. **非 assistant 恒 0.0**：system/user/pad 一律 0；空 assistant 行整行 0。

## 对拍口径
参考解「逐行扫描出极大段 → 段内 `base/count` 填充」。判题 `_oracle` 用**另一套结构**：先给每个 token 打「行内轮号」（遇到 assistant 段起点才自增），再统计每个轮号的 token 数，最后逐位 `base[轮号]/该轮数` 回填——「先标号统计、再除法」与参考解「先找段、段内除」路径不同、结果一致。另设「每轮求和==基数」「无 assistant 全 0」「turn_weights 越界回退」「多行独立编号」断言，`allclose(atol=1e-6)`。

## 复杂度
\(O(B L)\)。
