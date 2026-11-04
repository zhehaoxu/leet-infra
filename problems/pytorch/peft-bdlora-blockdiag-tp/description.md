# PEFT：BD-LoRA 张量并行块对角因子

## 题目描述

**BD-LoRA**（arXiv:2510.23346，NeurIPS 2025，Amazon，已合入 HF PEFT `use_bdlora`/`BdLoraConfig`）把部分 LoRA 因子约束成**块对角**，使张量并行服务**零通信**：列并行层要求 \(B\) 块对角、行并行层要求 \(A\) 块对角，每个块对角分片驻留一张 GPU，消除 all-gather / all-reduce。等价解释：BD-LoRA 就是在基座权重的每个分片上挂独立的 rank-\(r/N\) LoRA。块对角降低表达力，靠提高 \(r\) 做等参数量补偿。

本题复刻官方 PEFT 实现（`BlockDiagonalLinear` + `BdLoraLinearVariant`）的确定性账：

- **紧凑存储与前向**：因子不存 \((out, in)\) 满矩阵，而存 `nblocks` 个分片的堆叠 `(out_features, in_features // nblocks)`；前向 `x → (B, nb, m)`、`weight.view(nb, n, m)`、`einsum("bim,inm->bin")` 逐分片相乘再拼接，3D 输入保留前导维。整除性不满足抛 ValueError。
- **块对角展开**：`torch.block_diag(*torch.chunk(weight, nblocks, dim=0))` 把分片还原为满布局，非对角块严格为零。
- **目标分类**：`target_modules_bd_a` / `target_modules_bd_b` 子串匹配；同时命中两组抛 ValueError；`match_strict=True` 时都不命中抛 ValueError，`False` 时回落 plain LoRA。
- **初始化**：块对角 `lora_A` 用 `kaiming_uniform_(a=√5)`（fan_in 看分块的 \(in/nb\)），`lora_B` 全零（初始 \(\Delta W=0\)）。
- **合并账**：展开后 `delta = (B_bd @ A_bd) · scaling`（linear 层 `fan_in_fan_out=False` 不转置）；合并后的基座权重与未合并的两步前向**逐位一致**。

## 任务

- `mode="bd_forward"`：x、weight（紧凑分片）、nblocks 入参，返回块对角前向输出；整除不满足抛 ValueError。
- `mode="block_diag_matrix"`：weight、nblocks 入参，返回展开满矩阵。
- `mode="classify_target"`：target_name、patterns_a、patterns_b、match_strict 入参，返回 "a"/"b"/"plain"，非法组合抛 ValueError。
- `mode="init_factors"`：in_features、out_features、r、nblocks、role（"a"/"b"）、seed 入参，返回紧凑权重（A kaiming 由 seed 锚定、B 全零）；role 非法抛 ValueError。
- `mode="delta_weight"`：weight_a、weight_b、nblocks、scaling、fan_in_fan_out 入参，返回 \((out,in)\) 增量。
- `mode="layer_forward"`：x、base_weight、base_bias、weight_a、weight_b、nblocks、scaling 入参，返回 `base + lora_B(lora_A(x))·scaling`。

## 示例

seed 0/1/2/3，float64：

```
bd_forward(x (4,8), weight (6,4), nblocks=2) -> y (4,6)
  y[0,0] = -1.2027563104668666, y[-1,-1] = 2.3503342488855252
block_diag_matrix(weight (6,4), 2) -> bd (6,8), bd[0,0] = -0.31128955419003207
  bd_forward(x) == x @ bd^T；非对角块严格为零
classify_target("o_proj", ["o_proj"], ["q_proj"], True) == "a"
init_factors(8, 6, 4, 2, "a", seed=0) -> (4,4), wa[0,0] = 0.470053001806553
init_factors(8, 6, 4, 2, "b") -> (6,2) 全零
delta_weight(wa, wb (6,2), 2, scaling=2) -> (6,8), delta[0,0] = -0.44661549644151793
layer_forward(...) == F.linear(x, base_w + delta, base_b)（合并恒等）
```

## 约束

- `in_features`、`out_features`（及 r 侧维度）必须被 `nblocks` 整除，否则 ValueError。
- 块对角展开的非对角块**严格为零**（`abs().sum() == 0`），对角块与分片 `torch.equal`。
- `match_strict=True` 且目标不命中任何模式 → ValueError；同时命中 A/B 模式 → ValueError。
- `fan_in_fan_out=True` 时 delta 取转置（Conv1D 风格权重）。
- 合并恒等：`layer_forward(x) == F.linear(x, base_w + delta, base_b)`。
