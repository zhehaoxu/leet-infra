# BD-LoRA 实现笔记

## 出处与结构

- 官方 PEFT `src/peft/tuners/lora/variants.py`（`BlockDiagonalLinear` + `BdLoraLinearVariant`，`LoraConfig.use_bdlora=BdLoraConfig`）；论文 arXiv:2510.23346（NeurIPS 2025，Amazon）；vLLM 侧 PR #28136。
- `BlockDiagonalLinear.weight` 形状 `(out_features, in_features // nblocks)`——nblocks 个分片沿 dim=0 堆叠；前向 `einsum("bim,inm->bin")`，`first_dims` 保留（3D 输入 reshape(-1, in) 后还原）。
- `weight_as_blockdiagonal_matrix`：`torch.block_diag(*torch.chunk(weight, nblocks, dim=0))`；A 因子 (r, in/nb) → (r, in)，B 因子 (out, r/nb) → (out, r)。
- `BdLoraLinearVariant.init`：子串匹配 `pattern in target_name`；both → ValueError；`match_strict` 且 neither → ValueError。A 挂 `init_zero=False`（kaiming），B 挂 `init_zero=True`（全零）。
- merge：`_get_bdlora_delta_weight` 先把两因子 `weight_as_blockdiagonal_matrix` 展开再 `transpose(weight_B @ weight_A, fan_in_fan_out) * scaling`；CPU 上的 fp16/bf16 提升 float32。

## 判题口径与怪癖

- **紧凑权重形状是核心账**：A 是 `(r, in//nb)` 不是 `(r, in)`；B 是 `(out, r//nb)`。块对角展开后 `bd_forward(x) == x @ bd.T`。
- kaiming fan_in 看**转置前** weight 的 dim1 = `in//nb`（分块宽度），不是完整 in（同 808 LoRA-SB 经验）；a=√5；seed 锚定可复现。
- delta 的块对角结构保持：非对角块 < 1e-12；对角块 = 分片 `wb[i] @ wa[i]`（乘 scaling）——与满矩阵乘法差 ~2e-16（浮点序），性质断言用 allclose。
- 合并恒等 `layer_forward == F.linear(x, base_w + delta, base_b)` 是最强信号（m-lf-* 与 m-dl-* 变异全部经由它或锚值暴露）。
- `fan_in_fan_out=True` 分支：`(weight_a.t() @ weight_b.t()) * scaling == delta.t()`——测试直接对拍该恒等式。
- classify 的错误消息文案照抄官方（"matches both A and B block-diagonal patterns" / "matches neither A nor B"）。
- 整除校验在 `bd_forward` 与 `init_factors` 双处（官方在 `BlockDiagonalLinear.__init__`，判题域搬到两个入口）。
- einsum("bim,inm->bin") 与手工分块 `cat([x_i @ w_iᵀ])` 差 ~1e-16，锚值以 einsum 路径为准。
