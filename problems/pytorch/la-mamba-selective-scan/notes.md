# 详解：Mamba selective scan 账

## 出处

`state-spaces/mamba` 的 `mamba_ssm/ops/selective_scan_interface.py::selective_scan_ref`（2026-10-31 经 GitHub contents API 逐行取得）。这就是 Mamba 论文（2312.00752）S6 层的权威参考实现：`ΔA = exp(Δ·A)`（ZOH 一阶）、`ΔB_u = Δ·B·u`、`x_t = ΔA x_{t-1} + ΔB_u`、`y_t = C_t·x_t`。复数分支（`y.real * 2`）与 G 组重复（`repeat(B, "B G N L -> ...")`）不在本题范围，notes 记录剔除原因（复数的判题信号弱、G 组只多一步 repeat 接线）。

## 核心与易错点

1. **`delta` 预处理顺序**：先 `+ delta_bias` 再 `softplus`——顺序颠倒会被 `check_delta_bias_softplus` 的等价重排断言抓到（先 softplus 后加 bias ≠ 先加后 softplus，softplus(b)+Δ ≠ softplus(b+Δ)）。
2. **B/C 四种组合**：静态/变体各自独立判定（`B.dim() >= 3`）。变异「变体 B 用静态 einsum」在 B 变体下把 `B[b,n,l]` 换成 `B[d,n]`——当 dim==bN 且 N==L 时形状恰好不炸但数值错，靠随机对拍抓。
3. **`last_state` 是终态 `x`**（第 L-1 步），不是输出的别名；「减掉最后一步注入」的变异被手算锚点直接抓（`2ln2` 变 `2ln2 - ΔBu`）。
4. **dtype 账**：上游 `u.float()` 把一切拉到 float32 计算、最后 `.to(dtype_in)` 转回——float64 入参的输出 dtype 仍是 float64，但**计算精度是 float32**（与上游一致）；`A` 漏 float 化会让 einsum 直接 dtype 报错（测试抓到）。
5. **分块 carry 恒等**：把 L 切段后以段 1 终态为段 2 初态，与整条递推逐位一致——这是 Mamba 分块扫描（mamba_chunk_scan）正确性的数学基础；测试用 oracle 分段验证。solution 不提供初态入参（保持接口窄），分段等价由 oracle 承担。
6. **z 门控在 D 跳连之后**：`out = (y + u*D) * silu(z)`——先跳连后门控；顺序错变异被 z 用例抓。

## 变异测试（9/9 全抓）

exp 丢失、状态不更新（丢注入项）、变体 B 换静态 einsum、D 跳连删除、z 门去 silu、softplus/bias 换序、last_state 非终态、变体 C 换静态、dtype 转回删除。

## 复杂度

O(B·D·N·L) 递推，规模 ≤ 2×3×3×6，毫秒级。
