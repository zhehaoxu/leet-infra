# 详解：CTC 损失（前向算法）

## 出处
Graves, Fernandes, Gomez, Schmidhuber, *Connectionist Temporal Classification*（ICML 2006）。前向变量递推：扩展标签序列 \(\ell'\) 长 \(L'=2L+1\)（两端与间隙插 blank），\(\alpha_t(k)\) 为到时刻 \(t\)、停在扩展位置 \(k\) 的路径概率和，损失 \(-\log(\alpha_T(L'-1)+\alpha_T(L'-2))\)。

## 核心与易错点
1. **扩展标签 \(\ell'\)**：`label_at(k)` 偶数下标为 blank、奇数下标为 `labels[k//2]`。这是「可跳过 blank、可合并重复」的标准 trick。
2. **两条转移**：位置 \(k\) 每步可由 \(k-1\)（发射当前符号）或 \(k-2\)（跳过 blank/合并重复）到达；但 \(k-2\) 只有在「当前符号非 blank 且与 \(k-2\) 处符号不同」时才允许——这正是防止把 `a-a` 误当成两个 `a`。
3. **log 空间累加**：所有求和用稳定 `logaddexp`，`-inf` 表示不可达状态。用普通加法在长序列会下溢。
4. **终态合并**：结束时末位与次末位（blank）都可能是接受态，需 `logaddexp` 合并（\(L'=1\) 即空标签时只有全 blank 路径）。
5. **blank = V-1**：题目约定，不从外部传入。

## 对拍口径
参考解为纯前向 DP（`_lae` 稳定合并、`label_at` 扩展），返回 `-total` 标量。判题 `_oracle` **暴力枚举**全部 \(V^T\) 条路径，用 `collapse`（合并连续重复、删 blank）筛出等于目标标签者、累乘概率求和后取 `-log`——结构与 DP 完全不同，指数级但仅在极小 \(T,V\) 下运行，交叉验证前向算法。另设「labels=[0]」「labels=[0,1]」两个 example 数值锚点。

## 复杂度
前向 DP \(O(T L)\)（每步遍历 \(O(L)\) 个扩展位置）；对拍 oracle \(O(V^T\cdot T)\)。
