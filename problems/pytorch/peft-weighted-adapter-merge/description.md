# 加权 Adapter 合并（add_weighted_adapter）

PEFT 的 `LoraModel.add_weighted_adapter`（`peft/tuners/lora/model.py`）把多个已训练的 LoRA adapter 按权重合成一个新 adapter——部署前把任务算术、符号拆分、SVD 降秩一次性做掉的**离线合并账**。与 `peft-lora-merge`（单 adapter 折叠 W+sBA）和 `peft-ties-dare-merge`（论文级 TIES/DARE 算子）划界：本题考的是**库机制本身**——组合类型分派、rank 规则、scaling 归一、三种合并路径的精确公式。

## 机制拆解

### 1. rank 规则（`_check_add_weighted_adapter`）

- **单源强制 linear**：只给一个 adapter 时，无论传什么 combination_type 都改成 `linear`；
- **linear 族**（linear/ties/dare_ties/dare_linear/magnitude_prune）：所有源 rank 必须相等，否则 ValueError，new_rank = 该值；
- **cat**：new_rank = 各源 rank 之和（可能 OOM，官方文档明示）；
- **\*_svd 族**：new_rank = `svd_rank or max(ranks)`；
- 其它 combination_type → ValueError。
- 每个 adapter 的参与 rank 是 `config.r`，有 rank_pattern 时取 `max(r, *rank_pattern.values())`（分配上界防爆）。

### 2. scaling 归一

新 adapter 的配置被强制为 `r = lora_alpha = new_rank`、`use_rslora=False`、`alpha_pattern/rank_pattern={}`，即**新 adapter 的 scaling 精确等于 1**——每个源的 scaling 必须在合并时折进权重里（否则会被算两次）。若继承 use_rslora=True，scaling 会变成 r/√r=√r 而破坏这个不变量。

### 3. cat 分支（精确）

- lora_A 乘 `weight * scaling[adapter]`，lora_B **不缩放**；
- A 沿 dim=0（rank 维）拼接，B 沿 dim=1 拼接；
- 拼接后 `A_new @ B_new = Σᵢ wᵢ·sᵢ·BᵢAᵢ` 精确成立。

### 4. linear 分支（粗糙近似，√拆分 + 符号单侧）

`weight_with_scaling = weight * scaling[adapter]`，取 `sqrt(|·|)`，**符号只加在 A 侧**：

```
w_A = sqrt(|w·s|) · sign(w·s)，w_B = sqrt(|w·s|)
A_new = Σ w_A·Aᵢ，B_new = Σ w_B·Bᵢ
```

两个源时交叉项使 `A_new @ B_new ≠ Σ wᵢsᵢ·BᵢAᵢ`（官方文档明示 linear 是 rough approximation）；单源时精确。负权重若两侧都带符号会互相抵消——这就是符号单侧的原因。

### 5. svd 分支（精确，kohya 约定）

- 先按 `get_delta_weight` 组出完整 delta：`deltaᵢ = (Bᵢ @ Aᵢ) · sᵢ`（**scaling 已折入**）；
- 组合用**原始权重**：`combined = Σ wᵢ · deltaᵢ`（负权重自然生效）；
- SVD 后 **奇异值全部吸收进 B**：`B = U[:, :rank] @ diag(S[:rank])`，`A = Vh[:rank, :]`（kohya-ss sd-scripts 约定，不是 √ 对半分）；
- `svd_clamp`：对 `cat([U.flatten(), Vh.flatten()])` 取分位数 hi，U/Vh 对称 clamp 到 [−hi, hi]；
- 返回顺序 `(A, B) = (Vh, U)`。

### 6. bias 合并

`new_bias = Σ wᵢ · sᵢ · bᵢ`，只累计有 lora_bias 的源（scaling 显式乘回）；全 None 抛 ValueError。

## 判题域

单入口 `solve(mode, **kw)`，七个 mode：`effective_rank` / `resolve_rank` / `resolve_scaling` / `merge_cat` / `merge_linear` / `merge_svd` / `merge_bias`。float64 数值锚点；Embedding/Conv 分支与 TIES/DARE 族不在域内（TIES/DARE 见 `peft-ties-dare-merge`）。
