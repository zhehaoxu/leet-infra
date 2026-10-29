# 解题说明 · DeLoRA 的行范数归一化更新

## 出处

- `huggingface/peft` 的 `src/peft/tuners/delora/layer.py`：`_compute_delta` 给出 `delta = B @ diag(lam/r / (||A_i|| * ||B^j||)) @ A` 再按输入通道乘 `w_norm`；`forward` 用 `(x * w_norm) @ Aᵀ → * scaling → @ Bᵀ` 的等价逐元素路径；`reset_delora_parameters` 在训练前用 `torch.norm(w.data, dim=0).detach()` 固定 `w_norm`。
- 论文编号未逐一回源，公式以源码为准（notes 只承担口径说明，不宣称与论文逐字一致）。

## 核心与易错点

**rank 来自 `A` 的行数，不是 `B`。** 源码里 `A` 是 `(r, in_features)`、`B` 是 `(out_features, r)`，与「先降维再升维」的 LoRA 布局一致；反过来读会让 `dim=1` / `dim=0` 的范数取错轴，`scaling` 直接变成另一个形状。

**范数下限夹在范数上，不是平方范数上。** `clamp(norm, min=1e-4)` 挡住全零行/列；若先平方再夹、或者把下限写成 `1e-8`，零行场景的缩放会差几个数量级。本题用一个全零行把这条边界显式测出来。

**缩放与数据无关，但也不是常量。** `scaling_i` 只由当前 `A` 的行范数与 `B` 的列范数决定，不看输入激活；但训练中 `A`、`B` 在更新，所以 `forward` 每次重算（源码在 `forward` 里现算 `An`/`Bn`），不能在初始化时缓存。

**`w_norm` 按列（输入通道）乘。** 它来自基座权重 `w` 的 `dim=0` 范数，长度是 `in_features`；乘成 `unsqueeze(1)` 的行方向，结果整体错位。

**`‖delta‖_F = lam / sqrt(r)` 是有条件的恒等式。** 只有当 `A` 的行、`B` 的列各自是单位向量且互相正交时，各秩 1 分量正交、范数才精确可加；一般情况下只能保证 `‖delta‖_F <= lam`。题目同时检查「正交特例取等」与「任意因子不超界」。

**合并路径必须与 `forward` 逐元素路径一致。** `x @ deltaᵀ` 展开是 `((x * w_norm) @ Aᵀ * scaling) @ Bᵀ`；把 `w_norm` 放到 `A` 的左侧或 `B` 的右侧都会让两条路径对不上。

**精度。** 范数连乘再取倒数对 float32 很敏感，内部统一 float64、出口再转 float32，避免随机用例上的相对误差累积。

## 与其他题的分工

- `peft-dora`：DoRA 把权重拆成「列范数 × 方向」两段并只训方向；DeLoRA 是给低秩**更新量**加范数上界，不分解基座。
- `peft-lokr`、`peft-oft-cayley`：约束写在参数化里（正交/半正交因子）；本题约束写在输出更新的范数上，因子本身不要求正交。
- `peft-hra-householder-reflection`、`peft-road-block-rotation`：同批适配器，分别用反射乘积与分组旋转改权重；DeLoRA 只做加性低秩更新与缩放。
- 量化题里的 per-channel scale 是数值表示的一部分；本题的 `scaling` 是容量约束，二者不要混用同一套推导。
