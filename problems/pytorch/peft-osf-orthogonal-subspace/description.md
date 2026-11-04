# OSF 正交子空间微调

OSF（Orthogonal Subspace Fine-tuning）已合入 HF PEFT（`peft/tuners/osf/`），面向**持续学习**：用自适应 SVD 把权重拆成「冻结的高秩部分」（保留旧任务的重要方向）与「可训练的低秩部分」，训练时把梯度**投影到与高秩子空间正交**的方向上——既保全参表达力，又防灾难性遗忘，且不新增参数。

## 机制拆解

### 1. 权重分解

```
W ≈ U_high · diag(S_high) · V_high + U_low · diag(S_low) · V_low
```

- SVD 在 **float32 域**计算（`W.to(torch.float32)`，`full_matrices=False`），结果 cast 回原 dtype——恒等式容差需按 fp32 管线放 1e-5；
- `k = min(top_k, min_dim)`：前 k 个奇异分量冻结（high），其余可训练（low）；
- `rank_high = k`；`top_k` 越界时夹到 min_dim（此时 low 为空）。

### 2. 秩解析（effective_rank 语义）

**`effective_rank` 是保留（冻结）秩，不是可训练秩**——可训练秩 = min_dim − effective_rank，与 LoRA 的 r 相反：

| 输入 | 解析 |
|---|---|
| int | 直接作为保留秩 |
| float ∈ (0, 1] | `int(fraction · min_dim)` |
| None | `min_dim // 2`（默认 50%） |

float 出界（≤0 或 >1）抛 ValueError。

### 3. 梯度正交投影

```
dU ← dU − U_high @ (U_highᵀ @ dU)      # 对 U_high 列空间正交化
dV ← dV − dV @ (V_highᵀ @ V_high)      # 对 V_high 行空间正交化（右乘 Gram）
```

判题敏感点：

- **U 投影是左乘投影矩阵，V 投影是右乘 Gram**——两侧公式不对称（U: `U_high @ (U_highᵀ dU)`；V: `dV @ (V_highᵀ V_high)`）；
- 正交性仅在 **U_high 列正交 / V_high 行正交**（SVD 因子天然满足）时成立——test 必须用 QR 正交基构造，随机高斯会让正交性断言失效；
- 减号方向：加号变异把「去掉高秩分量」变成「放大两倍高秩分量」。

分布式 all-reduce 分支不在判题域。

### 4. 重构

`U_high diag(S_high) V_high + U_low diag(S_low) V_low` 精确还原 W（fp32 管线内）；单独的 high 部分即最优 rank-k 近似。

## 判题域

单入口 `solve(mode, **kw)`，五个 mode：`decompose` / `reconstruct` / `resolve_rank` / `project_u` / `project_v`。SVD 输入 W 由用例给定（确定性）；数值锚点 float64，fp32 管线处容差 1e-5。
