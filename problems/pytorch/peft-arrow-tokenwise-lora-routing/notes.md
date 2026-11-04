# Arrow 实现笔记

## 出处与结构

- 官方 PEFT `src/peft/tuners/lora/arrow.py`（`ArrowLoraLinearLayer`，PR #2644，与 GenKnowSub 同 PR）；论文 arXiv:2405.11157（NeurIPS 2024，Mila）。
- 原型 = ΔW=B@A 的 top 右奇异向量；幂迭代算子 `v ↦ Aᵀ(BᵀB)(Av)`——不物化 ΔW（out×in）或 M=Aᵀ(BᵀB)A（in×in）；fp32 迭代；`rng_seed` 造私有 Generator 决定 v₀。
- 路由：`sim = torch.abs(tok @ protos.T)`（sign-agnostic）；topk → 未选中填 -inf → `softmax(full_score / temperature)`；einsum 三连（z/y/delta）+ 逐适配器 scaling；dropout 在 delta_flat 之后（判题域恒等）。
- `on_adapter_change` 使 `_protos_ready` 失效惰性重建；`check_loaded_lora_compatibility_arrow` 校验 rank/形状/模块集。

## 判题口径与怪癖

- **幂迭代的收敛等价类**：初值扰动、±方向、迭代次数 ≥ 收敛步数都真等价（15 vs 14 diff=0，fp32 精度内收敛）；可抓的是**欠迭代**（iters=2 与 15 差 0.26）与算子/矩阵顺序变异。
- solve 调度层硬编码 `kw.get("iters", 15)` 会**遮蔽**函数签名的默认值——变异函数签名默认无效，变异点必须选在 solve 层（同 809 教训的另一面）。
- **fp32 → LoRA dtype 回投**是官方 build_prototypes 行为（`proto32.to(dtype=A.dtype)`）；判题域 fp64 时路由全程 fp64，dtype 断言跟随 A 而非 fp32。
- 相似度的 abs 是官方 sign-agnostic 设计；去掉 abs 是可抓变异（测试含负相似度 token）。
- -inf 填充使非选中 softmax 系数**严格为 0**（不是小数）；`torch.equal(coeff[~mask], 0)` 断言。
- equal-score tie（零 token 与所有原型相似度全 0）由 topk 索引序确定——测试用之验证均匀分裂 0.5/0.5。
- 兼容校验三个 ValueError（rank/shape/modules）必须**用例间互不兜底**：rank 用例的形状要一致，否则 rank 变异被 shape 检查兜底存活。
- 锚值容差：singular_vec 1e-7（fp32 迭代精度）、路由/组合 1e-9（fp64 线性代数）。
