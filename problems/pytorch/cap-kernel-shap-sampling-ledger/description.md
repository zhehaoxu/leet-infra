# Captum Kernel SHAP 采样与评估次数账

## 背景

归因方法要回答一句话：**模型这次输出，是哪些输入贡献的？**

Shapley 值给了一个漂亮但昂贵的答案：把每个特征「开」和「关」，枚举所有顺序，
看每个特征在多少种顺序下贡献了多少平均增量。数学上无懈可击，计算量却是
`M!` —— M 个特征就是 M 的阶乘次前向。M=10 时是 360 万次。

Kernel SHAP（arXiv **1705.07874**）给了一条工程出路：**用采样代替枚举**。
它的洞见是，Shapley 值的加权回归里，每个样本的权重恰好是一个可以闭式写出的
函数（Shapley kernel），于是可以按这个权重去采样少量样本，再解一个带权
最小二乘。

Captum 在 `captum/attr/_core/kernel_shap.py` 里实现了这条路。它是 **LIME 框架**
的一个特化配置 —— LIME 负责「采样 → 评估 → 拟合代理模型」这条流水线，
Kernel SHAP 换掉了其中的三块零件：

| 零件 | LIME 默认 | Kernel SHAP 换成 |
|---|---|---|
| 采样分布 | 每个特征独立 `bernoulli(0.5)` | Shapley kernel `p(k) = (M-1)/(k(M-k))` |
| 相似度核 | 余弦距离的高斯核 | 端点 `inf_weight` / 内部 `1.0` |
| 代理模型 | `SkLearnLasso(alpha=0.01)` | `SkLearnLinearRegression()` |

同时 `captum/attr/_core/shapley_value.py` 提供了另外两条 Shapley 路线
（采样版 `ShapleyValueSampling` 与精确版 `ShapleyValues`），它们共享同一套
「排列 → 逐步替换 → 累积贡献」的机制，但**前向评估次数的口径完全不同**。

本题要你用纯 Python 复现这套采样与计数规则的**精确数值口径**。
这条链路上有十来处「看起来合理但官方不是这么写」的细节，
本题的判题会把它们逐个钉死。

## 口径约定

- `num_interp_features`（记作 **M**）：可解释特征数，`>= 1`。
- `interpretable_sample`：长度 M 的 **0/1 向量**，第 i 位表示第 i 个可解释特征
  是否被「选中」。
- `rand_vals`：长度 M 的浮点向量（官方是 `torch.randn(1, M)`，本题作显式入参
  ⇒ 全部确定性）。
- 所有出参统一为 `float`（内部能走精确有理数的部分走 `fractions.Fraction`）。
- 非法输入一律抛 `ValueError`。

---

## mode 1：`select_distribution`

复现官方的采样分布构造：

```python
num_features_list = torch.arange(num_interp_features, dtype=torch.float)
denom = num_features_list * (num_interp_features - num_features_list)
probs = (num_interp_features - 1) / denom
probs[0] = 0.0
# ... num_select_distribution=Categorical(probs)
```

**签名**

```python
select_distribution(num_interp_features)
    → {"probs": [...], "denom": [...], "total": float,
       "normalized": [...] | None, "sampling_range": [lo, hi],
       "probs_f32": [...], "normalized_f32": [...] | None}
```

**要点**

1. ★ `num_features_list = arange(M)` ⇒ **k 的取值范围是 `0 .. M-1`**，
   **不含 M**。所以「全选」（k = M）这个端点**永远不会**被随机采样到。
2. ★ `denom = k * (M - k)`。**k=0 时分母为 0** ⇒ `(M-1)/0 = inf`，
   官方用 `probs[0] = 0.0` 把它**手动抹掉**。
3. ★ **k = M-1 时 `denom = (M-1) * 1 = M-1`** ⇒
   `probs = (M-1)/(M-1) = **1.0**`，**不是**无穷。
   也就是说「几乎全选」这个端点的权重是 1，
   而理论上它应该和 k=1 一样重（对称）。
4. ★ probs **不归一化**：M=5 时 probs 之和是 `10/3` 而不是 1。
   真正生效的分布是 `Categorical(probs)` 内部算的 `probs / probs.sum()`。
5. ★ **M=1 的退化**：`probs = [0.0]`，和为 0 ⇒ 归一化出 **NaN** ⇒
   `Categorical` 抛 `ValueError`（Simplex 约束失败）。
   **官方 Kernel SHAP 在只有一个可解释特征时直接崩。**
   此时 `normalized` 与 `normalized_f32` 都必须是 `None`。
6. ★ 官方全程 **float32**（`torch.arange(M, dtype=torch.float)`），
   所以 `probs_f32` 与精确有理数有 float32 精度级别的差异
   （M=16 时约 `4.3e-08`）。`probs` / `denom` / `total` / `normalized`
   走**精确有理数**口径，`probs_f32` / `normalized_f32` 走**官方 float32** 口径。
7. `sampling_range` 是**真正可能被采样到的 k 的范围**，即 `probs > 0` 的下标
   区间。M>=2 时是 `[1, M-1]`；M=1 时是空区间 `[0, -1]`。

**示例**

```python
solve("select_distribution", num_interp_features=5)["probs"]
# → [0.0, 1.0, 2/3, 2/3, 1.0]     总和 = 10/3
solve("select_distribution", num_interp_features=5)["normalized"]
# → [0.0, 0.3, 0.2, 0.2, 0.3]      总和 = 1
solve("select_distribution", num_interp_features=1)["normalized"]
# → None
```

---

## mode 2：`similarity_weight`

复现官方的相似度权重：

```python
num_selected_features = int(interpretable_sample.sum(dim=1).item())
num_features = kwargs["num_interp_features"]
if num_selected_features == 0 or num_selected_features == num_features:
    similarities = self.inf_weight
else:
    similarities = 1.0
return torch.tensor([similarities])
```

**签名**

```python
similarity_weight(interpretable_sample, num_interp_features, inf_weight=1e6)
    → {"weight": float, "num_selected": int, "is_endpoint": bool}
```

**要点**

1. ★ 端点（**全 0** 或 **全 1**）的权重是 `inf_weight = **1000000.0**`，
   其余样本权重一律是 `1.0`。
2. ★ 这个 `1000000.0` **不是真正的无穷**。官方注释写得很清楚：
   端点的理论权重应该是无穷大，「实践中只要让这个权重足够大就行，
   这样能强制训练出的线性模型满足**端点准则**」。
   所有其他权重都是 1，所以这个值同时是「相对量级」。
3. 判据是「**选中的特征数**」，与 1 出现在哪些位置**无关**。
4. `inf_weight` 可以由调用方指定（默认 `1000000.0`）。
5. `interpretable_sample` 必须是长度恰好 M 的 0/1 向量，否则抛 `ValueError`。

**示例**

```python
solve("similarity_weight", interpretable_sample=[1, 0, 0],
      num_interp_features=3)["weight"]     # → 1.0
solve("similarity_weight", interpretable_sample=[0, 0, 0],
      num_interp_features=3)["weight"]     # → 1000000.0
```

---

## mode 3：`perturb_sample`

复现官方扰动向量的生成：

```python
yield torch.ones(1, num_features, device=device, dtype=torch.long)
yield torch.zeros(1, num_features, device=device, dtype=torch.long)
while True:
    num_selected_features = kwargs["num_select_distribution"].sample()
    rand_vals = torch.randn(1, num_features)
    threshold = torch.kthvalue(
        rand_vals, num_features - num_selected_features
    ).values.item()
    yield (rand_vals > threshold).to(device=device).long()
```

**签名**

```python
perturb_sample(rand_vals, num_selected_features, num_interp_features)
    → {"sample": [...], "threshold": float, "num_selected": int,
       "endpoint": bool}

perturb_sample(kind="endpoints", num_interp_features)
    → {"samples": [[...], [...]], "num_endpoints": 2}

perturb_sample(kind="sequence", rand_vals_list, selected_counts,
               num_interp_features)
    → {"samples": [[...], ...], "num_endpoints": 2, "num_random": int}
```

**要点**

1. ★ 前两个样本是**固定端点**：**全 1**（原始输入）与**全 0**（baseline），
   **无条件**产出，不消耗随机预算。
   所以 `n_samples = 2` 时只有端点、没有随机样本。
2. ★ 阈值取的是 **第 `(M - k)` 小**的值
   （`kthvalue` 的 k 是 1-based 的「从小到大数的第几个」），
   **不是**第 k 大。
3. ★ 比较是**严格大于** `rand_vals > threshold`。
   无并列时正好选中 k 个；**并列时会少选** ——
   比如 `rand_vals = [5, 5, 5, 1]`、`k = 2`、`M = 4` 时
   阈值是第 2 小（也是 5.0），严格大于 5.0 的**一个都没有**，
   于是选出 **0** 个而不是 2 个。
4. `num_selected_features` 必须落在 `1 .. M-1`。
   `k = M` 时官方 `kthvalue(rand_vals, 0)` 的索引越界
   （`kthvalue` 的 k 是 1-based，0 不合法）⇒ 抛异常 ⇒ 你的实现应抛 `ValueError`。
5. `rand_vals` 的长度必须等于 M。
6. `kind="sequence"` 时，`rand_vals_list` 与 `selected_counts` 逐项对应，
   长度必须相同；输出序列 = `[全 1, 全 0] + 逐项生成的随机样本`。

**示例**

```python
solve("perturb_sample", rand_vals=[3.0, 2.0, 1.0],
      num_selected_features=1, num_interp_features=3)["sample"]
# → 阈值 = 第 (3-1)=2 小 = 2.0 ⇒ 严格大于 2.0 ⇒ [1, 0, 0]
solve("perturb_sample", rand_vals=[5.0, 5.0, 5.0, 1.0],
      num_selected_features=2, num_interp_features=4)["num_selected"]
# → 0（并列导致少选）
```

---

## mode 4：`n_evaluations`

复现两条 Shapley 路线的前向评估次数：

```python
# ShapleyValueSampling
def _get_n_evaluations(self, total_features, n_samples, perturbations_per_eval):
    return math.ceil(total_features / perturbations_per_eval) * n_samples

# ShapleyValues
def _get_n_evaluations(self, total_features, n_samples, perturbations_per_eval):
    return math.ceil(total_features / perturbations_per_eval) * math.factorial(
        total_features
    )
```

**签名**

```python
n_evaluations(total_features, n_samples, perturbations_per_eval, exact=False)
    → {"n_evaluations": int, "mode": "exact" | "sampling",
       "permutation_factor": int}
```

**要点**

1. 采样版是 `ceil(total_features / perturbations_per_eval) * n_samples`
   —— **先向上取整再乘**。
2. ★ 精确版（`ShapleyValues`）把 `n_samples` **完全忽略**，
   换成 `factorial(total_features)` —— 因为它枚举了全部 M! 个排列。
3. `perturbations_per_eval` 只出现在 `ceil` 的分母里，
   **不**影响乘法因子。
4. `mode` 出参标明走的是哪一套（`"sampling"` / `"exact"`），
   `permutation_factor` 是那个乘法因子本身。

---

## mode 5：`all_permutations`

复现官方的全排列枚举：

```python
def _all_perm_generator(num_features: int, num_samples: int) -> Iterable[Sequence[int]]:
    for perm in itertools.permutations(range(num_features)):
        yield perm
```

**签名**

```python
all_permutations(num_features, num_samples=None)
    → {"permutations": [[...], ...], "count": int, "num_samples_ignored": bool}
```

**要点**

1. ★ `num_samples` 参数被**完全忽略** —— 生成器产出全部 `M!` 个排列，
   与传进来的 `num_samples` 无关。
2. 排列顺序是 `itertools.permutations` 的字典序（第一项是升序 `[0, 1, ..., M-1]`）。
3. `count` = `M!`。

---

## 常见错误清单

1. `probs[0]` 留成 `inf`（没做 `probs[0] = 0.0` 的手动抹除）。
2. `probs` 提前归一化（官方返回的是**未归一化**的原始 probs）。
3. k 的取值范围写成 `0 .. M`（含 M）—— 官方是 `arange(M)`，**不含 M**。
4. `denom` 写成 `k * (M - k + 1)` 之类。
5. `probs[M-1]` 按「理论无穷」处理（实际是 **1.0**）。
6. M=1 时仍然返回一组归一化概率（官方会崩，`normalized` 必须是 `None`）。
7. `normalized_f32` 用了 double 算（官方是 **float32** 全程）。
8. 端点权重写成真正的 `inf`（官方是有限值 **1000000.0**）。
9. 端点判据只判「全 0」或只判「全 1」（官方是**两者都算**）。
10. `inf_weight` 参数被忽略。
11. 阈值取**第 k 大**而不是第 `(M - k)` 小。
12. 阈值比较用 `>=` 而不是**严格大于**。
13. 忘了并列时会**少选**（`[5,5,5,1]` + k=2 ⇒ 选 0 个）。
14. `k = M` 时不报错（官方 `kthvalue` 索引越界）。
15. 两个固定端点样本的顺序写反（全 1 在前、全 0 在后）。
16. 端点样本被放到序列**末尾**而不是最前。
17. 精确版的评估次数用 `n_samples` 而不是 `factorial(total)`。
18. `ceil` 写成 `floor`，或乘法与取整的**先后顺序**搞反。
19. `perturbations_per_eval` 也参与了乘法。
20. 全排列枚举被 `num_samples` 截断。

## 自检

```bash
python starter.py            # 骨架自检
```

提交前请确认：

- 五个 mode 的返回键名与题面完全一致。
- 所有出参是 `float`（不是 `Fraction`）。
- 非法输入抛 `ValueError`。
- M=1 时 `normalized` 与 `normalized_f32` 都是 `None`。
