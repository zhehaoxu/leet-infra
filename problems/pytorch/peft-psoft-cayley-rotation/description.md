# PSOFT 因子间 Cayley 正交旋转

PSOFT 已合入 HF PEFT（`peft/tuners/psoft/layer.py`）。它对基座权重 W 做 SVD 并缓存 A/B 因子，在 A 与 B 之间放一个**可学习的 r×r 正交旋转 R**：ΔW = scaling · B(R − I)A。初始化 R = I（上三角参数全零 → Q = 0 → Cayley 恒等），训练起点增量为零。

## 机制拆解

### 1. 反对称参数化

`orth=True` 时只存 `size(size−1)/2` 个上三角参数，`index_put` 到上三角后 `Q − Qᵀ` 得到反对称矩阵 Q（对角恒零，Q[rows[k], cols[k]] = w[k]）。

### 2. Cayley 变换

```
R = (I + Q)(I − Q)⁻¹
```

官方用 `torch.linalg.solve(I − Q, I + Q, left=False)`——`left=False` 解 `X·A = B`，即 `X = B·A⁻¹`。Q 与其有理函数可交换，故 `left=True`（(I−Q)⁻¹(I+Q)）数值等价。R 是真旋转（det=+1）且正交（R·Rᵀ=I）。零参数 ⇒ R = I。

**Cayley-Neumann 级数**（避免求解器）：

```
R = I + 2Q + 2Q² + 2Q³ + … + 2Q^(t−2) + Q^(t−1)
```

系数规则：第 1 到第 t−2 项系数 2.0，**最后一项系数 1.0**（截断级数一般不正交）。可选 Frobenius 投影：`norm(Q) > eps` 时 `Q ← Q · eps/(norm + 1e-12)`（+1e-12 在 norm>0.9 约束下数值上不可分，属真等价域）。

### 3. 幅度向量

```
R = diag(vector_b) @ R @ diag(vector_a)    # 行缩放 vector_b[:, None] * R，列缩放 R * vector_a[None, :]
```

初始化 vector_a/vector_b 全 1（恒等）。

### 4. SVD 因子缓存与 A/B 初始化

full 模式 `torch.linalg.svd(W)`（固定输入确定性，判题域合法）：`Vr = U[:, :r]`、`Sr = S[:r]`、`Uhr = Vh[:r, :]`。

- `psoft_init`：A = Uhr（原始奇异向量），B = Vr·diag(Sr/scaling)（**奇异值整体进 B**）；
- `pissa_init`：A = diag(√(Sr/scaling))·Uhr，B = Vr·diag(√(Sr/scaling))（√ 分布）。

两种 init 的 B@A 相同（都等于截断 W/scaling）。未知 init 抛 ValueError。

### 5. 增量与前向

```
ΔW = scaling · B (R − I) A          # R = I ⇒ ΔW = 0
xa = x @ Aᵀ;  xr = xa @ Rᵀ
branch = (xr − xa) @ Bᵀ · scaling    # 与 x @ ΔWᵀ 一致
```

r 校验：`r > min(in, out)` 抛 ValueError。

## 判题域

单入口 `solve(mode, **kw)`，八个 mode：`skew_symmetric` / `orth_matrix` / `magnify` / `svd_factors` / `init_ab` / `delta` / `branch` / `r_check`。SVD 输入 W 由用例给定（full 模式确定性），判题口径问题就此解决；数值锚点 float64。
