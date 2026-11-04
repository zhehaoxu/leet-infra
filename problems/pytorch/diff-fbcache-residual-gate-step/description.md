# 扩散：First Block Cache 残差门控步（首块残差相对 L1 决策）

## 题意

First Block Cache（FBCache，ParaAttention）让扩散 Transformer 在相邻去噪步变化不大时**跳过首块之后的全部块**。实现 `solve(head_input, head_output, tail_output, state=None, threshold=0.05)`：

```text
residual = head_output − head_input                   # 首块残差
diff     = absmean(residual − prev_residual) / absmean(prev_residual)
compute  <=>  prev_residual is None or diff > threshold   # 首步强制 compute
```

- **compute 路径**：全部块真实执行，`final_output = tail_output`；state 更新 `head_block_output = head_output`、`head_block_residual = residual`、`tail_block_residuals = tail_output − head_output`。
- **cache 路径**（`diff <= threshold`）：中间块透传，`final_output = tail_block_residuals(缓存) + head_output`；state 三项**全部保持旧值**（连续 cache 步都与上次 compute 的残差比较——stale 比较）。

返回 `dict(final_output, computed, diff, state)`，state 为新 dict（纯函数式）。

## 关键语义

- **分子分母都是 absmean（绝对值均值）**，不是 mean、不是 max：残差差可能整体为负（残差变小），abs 之前 diff 为负会误判 cache。
- **stale 比较**：cache 步不更新任何 state 项；连续多步 cache 时 diff 的参照始终是**上次 compute 步**的残差。
- **cache 的输出公式**：`缓存尾残差 + 当前首块输出`——不是 `+ head_input`（首块输出≠输入），也不是当前 tail_output。
- **零分母规范化**：`absmean(prev) == 0` 时官方产生 0/0=nan（nan>x 为 False → cache）或 x/0=inf（→ compute）；判题域显式规范化为 0.0/inf 同语义。
- **首步/官方 reset 后**：`head_block_residual is None` 强制 compute 并重建尾残差；cache 决策但 `tail_block_residuals is None` 的不一致组合显式拒绝（官方会 TypeError）。

## 判题域约定

- 官方 hook 挂在 head/中间/尾块共享 StateManager；判题域把一次前向压缩为 `(head_input, head_output, tail_output)` 三个张量 + 显式 state——中间块的透传在 cache 路径下是恒等映射，数学等价。
- 官方支持 tuple 双输出（hidden_states/encoder_hidden_states）；判题域单张量。
- `threshold ≥ 0` 为判题域显式校验（官方默认 0.05 无校验）。

## 例子

prev 残差 `(0.5, 0.5)`、当前残差 `(0.75, 0.75)`：num = absmean(0.25, 0.25) = 0.25、den = 0.5、diff = 0.5 精确；threshold=0.5 时恰在边界（cache，严格 >）；残差再加 2⁻⁴⁰ 即 compute。残差 `(0.15, 0.05)` 对 prev `(0.2, 0.1)`：差 `(−0.05, −0.05)`，absmean 后 diff = 0.05/0.15 = 1/3 → compute（无 abs 的实现会得到 −1/3 误判 cache）。

数据规模：张量 `(1, 2)`–`(2, 2)` float64，threshold ∈ [0, 1]。
