# 详解：Tuned Lens 仿射探针账

## 出处

AlignmentResearch/tuned-lens（论文 2303.08112）：每个探针是「LayerNorm + 仿射」把第 ℓ 层残差流映射到词表空间；本题取其**仿射部分**（`logits_ℓ = W_ℓ h_ℓ + b_ℓ`，LayerNorm 归 notes 简化点——判题需要 LN 参数也入参，为控接口不再扩）。KL/CE 全 float64。

## 核心与易错点

1. **KL 方向**：`KL(p_final ‖ p_layer)`——p_final 是权重（`p·(ln p − ln q)`）。反向（`q·(ln q − ln p)`）被 `anchor_hand` 手算值与随机对拍同时抓。
2. **温度作用于双方**：`logits/T` 后 softmax；只作用一侧的变异被 `temperature_effect`（T=1 vs T=4 的 KL 与 CE 都必须变）抓。T ≤ 0 抛 ValueError。
3. **CE 的 target** 是 `argmax(final_logits)`（每 batch 一个），CE 账形状 [L,B]——用 `gather` 按 target 取；初版直接 `logp[:, arange(B), target]` 把 L 维当 batch 索引（shape 错），改 gather。
4. **KL=0 的条件**：只有「隐态经探针后的 logits 与 final 完全相同」的层（probe=LM head 时的最后一层）；更早层 KL>0——这正是 tuned lens 度量的层间差。初版用例误以为 probe=LM head 则全层 KL=0。
5. **`torch.matrix_rank` 在 torch 2.14 已迁到 `torch.linalg.matrix_rank`**（参数 `symmetric`→`hermitian`，旧入口直接报错）——`probe_rank` 账用新 API；秩亏探针（重复行）与满秩探针的秩差被 `probe_rank` 钉死。
6. **_py 的 float32 截断**：对拍工具把 float64 转成 float32，1e-9 级断言会被舍入噪声卡住——锚点容差需 ≥1e-6（`ce_monotone` 用例踩过）。

## 变异测试（7/7 全抓）

探针去 bias、KL 方向反转、温度丢失、CE 丢负号、argmax_match 反转、秩 tol 爆改、match 只看最后一层。

## 复杂度

O(L·B·V·D) 投影 + O(L·B·V) 分布账，规模 ≤ 4×2×9×6，毫秒级。
