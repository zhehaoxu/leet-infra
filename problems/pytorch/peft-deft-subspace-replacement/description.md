# DeFT 解耦子空间替换微调

DeFT（Decoupled Fine-Tuning）已合入 HF PEFT（`peft/tuners/deft/layer.py`）。与 LoRA 的「低秩增量」不同，DeFT 做**子空间替换**：把权重在给定方向上的投影去掉，再注入一个可学习的秩 r 校正—— adapted = (I − P_proj)·W + Q_P·R，即

```
delta = Q_P @ (R − rightᵀ @ W)          # 完整 DeFT
delta = −Q_P @ (rightᵀ @ W)             # PaRa（para=True）：纯子空间移除
```

其中 (Q_P, right) 来自投影分解，P_proj = Q_P @ rightᵀ。

## 机制拆解

### 1. 投影分解（fp32 域）

| method | Q_P | right | P_proj | 性质 |
|---|---|---|---|---|
| `relu` | P | relu(P) | P @ relu(P)ᵀ | 非正交 |
| `qr` | qr(P) 的 Q | Q_P（同一矩阵） | Q_P @ Q_Pᵀ | 正交、幂等 |

投影数学全程在 **float32** 下计算（半精度下 QR 不可靠），调用方再 cast 回。

### 2. 形状与秩

- P: (out_features, r)；R: (r, in_features)（para 模式无 R）；
- 有效秩封顶：`r = min(r, out_features)`（投影秩不超过 out_features）；r ≤ 0 抛 ValueError；
- scaling = alpha / r（类 LoRA α/r），无 alpha 时 1.0。

### 3. 恒等初始化

`R_init = rightᵀ @ W / scaling`，使得 `scaling · R_init == rightᵀ @ W`，初始 delta **精确为零**——训练从预训练权重出发学习注入，避免「开局移除子空间」导致的立即遗忘（`init_weights=False` 时 R 走 normal(0, 0.02·init_scale)，非恒等开局）。

### 4. merge 因子缓存

`M = rightᵀ @ W`（r × in，约 r/out_features 的内存）是 delta 中唯一依赖基座权重的部分；merge 缓存 M，unmerge 从 M 精确重构 delta（`Q_P @ (R·scaling − M)`）。delta 恒等式：`W + delta == (I − P_proj)W + Q_P R`。

### 5. 激活域前向

不物化 W + delta，直接在激活上算：

```
result = base_product − (base_product @ right) @ Q_Pᵀ + F.linear(x, R·scaling) @ Q_Pᵀ
```

其中 base_product = x @ Wᵀ（去 bias 的基座积）。**branch 返回的是完整输出**（base_product + 增量），恒等式 `out == base_product + x @ deltaᵀ`；恒等初始化下 out == base_product（适配器无感）。

## 判题域

单入口 `solve(mode, **kw)`，七个 mode：`resolve_r` / `resolve_scaling` / `project` / `merge_factor` / `delta_from_factor` / `init_R` / `branch`。fp32 管线处容差 1e-5；数值锚点 float64。
