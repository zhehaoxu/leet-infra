# 解题笔记 — PSOFT

## 官方账

- HF PEFT `peft/tuners/psoft/layer.py`（OrthLayer + PsoftLayer）；判题域取纯函数部分。
- **判题口径问题的解法**：之前顾虑「SVD 对数据的 oracle 独立性」——实际上 SVD 的输入 W 由用例直接给定，full 模式 `torch.linalg.svd` 完全确定性，再配正交性/恒等性断言（R·Rᵀ=I、det=+1、R=I ⇒ 零增量、branch ≡ x@ΔWᵀ）抓变异。PSOFT 可做。

## 踩坑

- **占位锚值写 `< 0.0`**：恒假断言（abs ≥ 0），第一版两处直接挂——anchor 必须是真值。
- **left=False 语义**：`solve(A, B, left=False)` 解 X·A=B（X=B·A⁻¹），不是常规的 A·X=B。Cayley 的 (I+Q)(I−Q)⁻¹ 用 left=False；left=True 数值等价（Q 的有理函数可交换），属真等价剔除。
- **Neumann 级数尾项**：第 1..t−2 项系数 2.0，最后一项（Q^(t−1)）系数 1.0——`R.add_(Q_power)` 不带 alpha。t 的分支嵌套（terms>1/2/3 三层）容易数错项。
- **投影手算**：solution 的 Q 来自 weight=w·100，test 手算用 Q=skew(w) 时要乘 100——线性性 skew(c·w)=c·skew(w) 成立但容易漏。
- **真等价三类**（剔除）：Cayley left 方向（可交换）、投影 `>` vs `>=`（浮点边界不可稳定构造）、+1e-12 护栏（norm>0.9 约束下相对差 <1.2e-12）。
- SVD 符号歧义不是问题（固定输入下 torch.linalg.svd 确定性），但锚值必须从真值取而不是手算猜测。

## mut 14 变异清单

skew：Q−Qᵀ→Q+Qᵀ、上三角→下三角索引；orth：solve 方向符号翻转、Neumann 首项系数、尾项系数、Q² 项退化；magnify：行/列缩放向量互换；svd：截断头→尾；init：psoft B 丢 scaling、pissa 丢 sqrt；delta：R−I→I−R（反号）、scaling 乘→除；branch：(xr−xa) 反号、Rᵀ→R。14/14 全 CAUGHT。
