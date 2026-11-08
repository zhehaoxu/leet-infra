# Captum 源码解剖：Kernel SHAP 采样与评估次数账

> 官方仓库：`pytorch/captum`（分支 `main`）
> 涉及文件：`captum/attr/_core/kernel_shap.py`、`captum/attr/_core/shapley_value.py`

## 1. 这套规则在 Captum 里的位置

Captum 的归因方法分成两大类：**白盒传播**（Integrated Gradients、DeepLIFT、
Layer Conductance 之类，直接从网络的中间量往回推）与**黑盒采样**
（LIME、Kernel SHAP、Shapley Values 之类，靠反复前向 + 拟合代理模型）。

本轮 847 做的是 DeepLIFT（白盒乘法传播规则），本轮 849 做的是 Kernel SHAP
（黑盒采样）。两者的机制完全不重叠：

| | DeepLIFT（847） | Kernel SHAP（849） |
|---|---|---|
| 核心 | 逐层传播规则（`nonlinear` / `softmax` / `maxpool`） | 采样分布 + 相似度权重 + 代理模型拟合 |
| 成本 | 与输入规模线性相关 | 与 `n_samples` 相关 |
| 关键 quirk | eps 严格小于、全局归一化、unpool 后写覆盖 | `probs[0]=0` 手动抹除、float32 精度、严格大于导致并列少选 |

Kernel SHAP 是 **LIME 框架**的一个特化配置。LIME 负责整条流水线：

```
采样（perturb_func） → 评估 forward_func → 算相似度（similarity_func）
      → 组 TensorDataset → interpretable_model.fit() → representation()
```

Kernel SHAP 换掉了其中三块零件（官方 `KernelShap.__init__`）：

```python
Lime.__init__(
    self,
    forward_func,
    interpretable_model=SkLearnLinearRegression(),
    similarity_func=self.kernel_shap_similarity_kernel,
    perturb_func=self.kernel_shap_perturb_generator,
)
self.inf_weight = 1000000.0
```

`captum/attr/_core/shapley_value.py` 则是另外两条路线（采样版
`ShapleyValueSampling` 与精确版 `ShapleyValues`）。

---

## 2. `probs`：Shapley kernel 的采样分布

官方代码（`kernel_shap.py::KernelShap.attribute` 的尾部）：

```python
num_features_list = torch.arange(num_interp_features, dtype=torch.float)
denom = num_features_list * (num_interp_features - num_features_list)
probs = (num_interp_features - 1) / denom
probs[0] = 0.0
return self._attribute_kwargs(
    ...,
    num_select_distribution=Categorical(probs),
    ...
)
```

### ★ 要点 2.1：k 的取值范围是 `0 .. M-1`（不含 M）

`torch.arange(M)` 产出 `0, 1, ..., M-1`。所以：

- **k=0** 有位置（但概率被抹成 0）；
- **k=M 不在范围内** ⇒ 「全选」这个端点**永远不会被随机采样到**。

实测（M=5）采样到的 k 集合 = `{1, 2, 3, 4}`。

### ★ 要点 2.2：`probs[0] = 0.0` 是**手动抹除 inf**

k=0 时 `denom = 0 * M = 0` ⇒ `(M-1)/0 = inf`。官方用一行
`probs[0] = 0.0` 把它盖掉。

这不是「数学上正确」的做法 —— 理论上 k=0 的 Shapley kernel 权重也是无穷大
（和 k=M 对称）。官方选择**直接去掉 k=0 这个样本**，而不是给一个有限权重。

### ★ 要点 2.3：k = M-1 时 `probs` 恰为 **1.0**

`denom = (M-1) * (M - (M-1)) = (M-1) * 1 = M-1`
⇒ `probs = (M-1)/(M-1) = 1.0`。

实测（M=5）：`probs = [0.0, 1.0, 0.6667, 0.6667, 1.0]`。
注意 k=1 与 k=M-1 的权重都是 1.0（对称），中间的 k 更小 —— 这就是
Shapley kernel 的形状。

### ★ 要点 2.4：`probs` **不归一化**

M=5 时 `sum(probs) = 10/3 ≈ 3.3333`，不是 1。
M=8 时是 `4.5375`。

真正生效的分布是 `Categorical(probs)` 内部算的 `probs / probs.sum()`。
实测 M=8 的归一化结果：

```
[0.0, 0.220386, 0.128558, 0.102847, 0.096419,
 0.102847, 0.128558, 0.220386]
```

### ★ 要点 2.5：官方全程 **float32**

`torch.arange(num_interp_features, dtype=torch.float)` —— 注意是
`torch.float`（**float32**），不是 float64。

所以官方 `probs` 带 float32 精度误差。实测 M=16 时，官方 float32 值与
精确有理数值的最大差是 `4.257e-08`。

### ★ 要点 2.6：M=1 直接崩

M=1 时 `probs = [0.0]`（分母 0，被抹成 0），`sum = 0`
⇒ `Categorical` 归一化出 **NaN** ⇒ 抛：

```
ValueError: Expected parameter probs (Tensor of shape (1,)) of distribution
Categorical(probs: tensor([nan])) to satisfy the constraint Simplex(),
but found invalid values: tensor([nan])
```

**官方 Kernel SHAP 在只有一个可解释特征时直接崩。**

---

## 3. `kernel_shap_similarity_kernel`：端点权重

```python
def kernel_shap_similarity_kernel(self, _, __, interpretable_sample, **kwargs):
    assert "num_interp_features" in kwargs, ...
    num_selected_features = int(interpretable_sample.sum(dim=1).item())
    num_features = kwargs["num_interp_features"]
    if num_selected_features == 0 or num_selected_features == num_features:
        similarities = self.inf_weight
    else:
        similarities = 1.0
    return torch.tensor([similarities])
```

### ★ 要点 3.1：端点权重是 `1000000.0`，**不是无穷**

官方注释说得很清楚：

```
# weight should be theoretically infinite when
# num_selected_features = 0 or num_features
# enforcing that trained linear model must satisfy
# end-point criteria. In practice, it is sufficient to
# make this weight substantially larger so setting this
# weight to 1000000 (all other weights are 1).
```

这个权重会进入代理模型的**加权最小二乘**：
端点（原始输入 / baseline）的输出必须被线性模型精确拟合，
所以给它一个压倒性的权重。

### ★ 要点 3.2：判据是「选中个数」，与位置无关

实测：M=3 时 k=0 与 k=3 的权重都是 `1000000.0`，
k=1 与 k=2 都是 `1.0`。

---

## 4. `kernel_shap_perturb_generator`：扰动生成

```python
def kernel_shap_perturb_generator(self, original_inp, **kwargs):
    ...
    num_features = kwargs["num_interp_features"]
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

### ★ 要点 4.1：前两个样本是**固定端点**

`yield ones`（原始输入）与 `yield zeros`（baseline）**无条件**产出，
不消耗随机预算。实测第一个样本是 `[[1,1,1,1]]`，第二个是 `[[0,0,0,0]]`，
dtype 是 `torch.int64`。

⇒ **`n_samples = 2` 时只有端点，一个随机样本都没有。**

### ★ 要点 4.2：阈值是**第 (M-k) 小**

`torch.kthvalue(rand_vals, num_features - num_selected_features)`。
`kthvalue` 的 k 是 **1-based** 的「从小到大数的第几个」。
所以传 `M-k` 取到的是**第 (M-k) 小**的值 —— 正好是「选出最大的 k 个」
所需的分界点。

### ★ 要点 4.3：比较是**严格大于** ⇒ 并列时**少选**

`rand_vals > threshold`。无并列时正好 k 个。

**但如果有并列**：`rand_vals = [5, 5, 5, 1]`、k=2、M=4
⇒ 阈值 = 第 2 小 = 5.0 ⇒ 严格大于 5.0 的**一个都没有**
⇒ 选出 **0** 个而不是 2 个。

实测对照（`rand_vals` 6 个、k=2）：

| 比较 | 选中个数 |
|---|---|
| `rand_vals > threshold` | 2 |
| `rand_vals >= threshold` | 3 |

### ★ 要点 4.4：`k = M` 时官方抛异常

`kthvalue(rand_vals, 0)` 的 k=0 越界：

```
RuntimeError: kthvalue(): selected number k out of range for dimension 1
```

实测 k=0 与 k=M+1 都抛这个错。

（实践中 k=M 不会发生，因为 `p(M) = 0`。但它是**域约束**的一部分。）

---

## 5. `_get_n_evaluations`：两套评估次数口径

```python
# shapley_value.py::ShapleyValueSampling
def _get_n_evaluations(self, total_features, n_samples, perturbations_per_eval):
    return math.ceil(total_features / perturbations_per_eval) * n_samples

# shapley_value.py::ShapleyValues
def _get_n_evaluations(self, total_features, n_samples, perturbations_per_eval):
    return math.ceil(total_features / perturbations_per_eval) * math.factorial(
        total_features
    )
```

### ★ 要点 5.1：采样版 = `ceil(total / ppe) * n_samples`

实测（total=5, n_samples=4, ppe=2）→ `ceil(5/2) * 4 = 3 * 4 = 12`。

`ceil` 在**前**，`* n_samples` 在**后**。写成 `ceil(total * n_samples / ppe)`
在 total=5/n=4/ppe=2 时是 `ceil(10) = 10`，与正确的 12 不同。

### ★ 要点 5.2：精确版**完全忽略 `n_samples`**

`ShapleyValues` 枚举全部 `M!` 个排列，所以乘法因子是 `factorial(total)`。
实测：

| total | n_samples | 结果 |
|---|---|---|
| 3 | 4 | 18 |
| 3 | 25 | 18 |
| 5 | 4 | 600 |
| 5 | 25 | 600 |

**`n_samples` 传什么都一样。**

### ★ 要点 5.3：`ppe` 只出现在 `ceil` 的分母里

`perturbations_per_eval` 是「一次前向并行评估多少个扰动」，
它只影响需要几批，不影响每批的规模。所以它**不**参与乘法。

---

## 6. `_all_perm_generator`：`num_samples` 被完全忽略

```python
def _all_perm_generator(num_features: int, num_samples: int) -> Iterable[Sequence[int]]:
    for perm in itertools.permutations(range(num_features)):
        yield perm
```

### ★ 要点 6.1：`num_samples` 是死参数

实测 `_all_perm_generator(3, 10)` 产出 **6** 个（= 3!），
`_all_perm_generator(5, 3)` 产出 **120** 个（= 5!）——
传进去的 `num_samples` 完全没有作用。

这是一个典型的「签名保留了但实现不用」的痕迹：上游把
`num_samples` 传进来是为了和 `_perm_generator`（随机版，**真的**用
`num_samples`）保持签名一致。

### ★ 要点 6.2：顺序是 `itertools.permutations` 的字典序

第一项是升序 `[0, 1, ..., M-1]`。

---

## 7. 官方源码的三个复现坑

### 坑 1：`probs` 未归一化，容易被「顺手归一化」

官方返回的 `probs` 之和是 `10/3`（M=5），不是 1。
如果你在实现里「顺手」归一化了，后面的 `Categorical(probs)` 就会
**二次归一化** —— 结果虽然碰巧一样（归一化幂等），但 `probs` 这个出参
就与官方不一致了。

验证手法：直接断言 `total == 10/3`（M=5）。

### 坑 2：float32 vs 精确有理数

官方全程 float32，参考解走 Fraction。M=16 时差 `4.3e-08`。
判题必须**同时**给出两个口径（`probs` 精确 / `probs_f32` float32），
否则要么判不过官方，要么判不过「按题面写精确有理数」的提交。

### 坑 3：`kthvalue` 的 k 是 1-based，且 0 越界

写 `sorted(vals)[M-k-1]`（0-based）而不是 `sorted(vals)[M-k]`。
而 `k = M` 时 `M-k = 0`，`sorted(vals)[-1]` 会**静默取到最大值**而不报错 ——
官方是抛 `RuntimeError`。这是一个「Python 负索引掩盖了官方错误」的典型例子。

---

## 8. 三层验证结构

1. **官方层**：从 upstream `kernel_shap.py` / `shapley_value.py` 用
   `ast.get_source_segment` **逐字抽取** `kernel_shap_similarity_kernel` /
   `kernel_shap_perturb_generator` / `_all_perm_generator` /
   `_perm_generator` / 两个 `_get_n_evaluations`，
   只剥离 `captum` 整包依赖（评测环境没有 `sklearn` / `tqdm`）。
2. **参考解层**：`/tmp/captum849_solution_staged.py`（纯 Python + `Fraction`）。
3. **oracle 层**：不依赖参考解、按官方算式独立重写一遍的纯 float 实现。

实测 **132 个用例**：
- `official-vs-ref ≤ 5.960e-08`（float32 精度，由 `probs_f32` 口径承担）；
- `oracle-vs-ref = 0.000e+00`（精确有理数口径完全一致）。

**为什么要分两个精度口径**：官方是 float32，题面如果只要求精确有理数，
那提交的答案与官方对不上；如果只要求 float32，那「精确值」这个更严格的口径
就没被钉住。两个都出，才能同时覆盖。

---

## 9. mutation 的三条原则

本轮的 mutation 清单共 43 项，最终 **43/43 CAUGHT**。踩过的坑：

1. **等价变异要主动剔除**。首轮 42 项里 2 个存活，逐一核对后确认是
   **真等价**：
   - 把 `probs[0] = Fraction(0)` 改成 `pass` —— 因为前面的
     `d == 0` 分支已经 append 了 `Fraction(0)`，**数值完全相同**；
   - 把 `nf = [f32(k) for k in range(M)]` 改成 `nf = [float(k) ...]` ——
     因为后续每个运算都被 `f32()` 包裹，**nf 的精度差异被完全吸收**。

   这两条都是「**在全部合法输入域上输出相同**」，判据是全域等价，
   不是「这一组用例上相同」。

2. **校验层必须拆成互不依赖的独立防线**。例如 `perturb_sample` 那条
   check 里同时写了：
   - 断言 `sample == oracle`；
   - 断言「阈值取第 k 大」的候选**必须不同**；
   - 断言「用 `>=` 代替 `>`」的候选**必须不同**；
   - 断言并列时 `num_selected == 0`。

   只写第一条的话，一个把 `>=` 写成 `>` 的实现可能「恰好」在无并列的
   用例上通过。

3. **不可观测的差异要设计专门的用例，且锚点必须落在真正会变的输入上**。
   本轮唯一的用例缺口出现在**双向断言的位置**上：起初把「第 k 大 vs
   第 (M-k) 小」的双向断言写在循环里的每个用例上，但有些数据的
   第 k 大恰好等于第 (M-k) 小（例如 `[3,2,1]`、k=2、M=3 时两者都是 2.0），
   断言反而**误报**。修法是把它挪到循环外，用一组**专门构造的非对称数据**
   `[9,1,7,3,5]`、k=2、M=5（第 2 大 = 7.0，第 3 小 = 5.0，不同）。

   教训：**双向断言本身也要挑数据** —— 它要求两个候选在**该数据上**不同，
   而不是一个「在所有数据上都不同」的普适性质。

---

## 10. 官方源码位置索引

| 符号 | 位置 |
|---|---|
| `KernelShap.__init__` 的 `inf_weight` | `kernel_shap.py:31-46` |
| `KernelShap.attribute` 的 probs 构造 | `kernel_shap.py:276-286` |
| `num_features_list = torch.arange(...)` | `kernel_shap.py:280` |
| `probs = (num_interp_features - 1) / denom` | `kernel_shap.py:284` |
| `probs[0] = 0.0` | `kernel_shap.py:286` |
| `Categorical(probs)` | `kernel_shap.py:296` |
| `kernel_shap_similarity_kernel` | `kernel_shap.py:307-330` |
| `kernel_shap_perturb_generator` | `kernel_shap.py:332-373` |
| `yield torch.ones(...)` / `yield torch.zeros(...)` | `kernel_shap.py:365-366` |
| `torch.kthvalue(rand_vals, num_features - num_selected_features)` | `kernel_shap.py:370-372` |
| `(rand_vals > threshold)` | `kernel_shap.py:373` |
| `_all_perm_generator` | `shapley_value.py:35-37` |
| `_perm_generator` | `shapley_value.py:40-42` |
| `ShapleyValueSampling._get_n_evaluations` | `shapley_value.py:582-584` |
| `ShapleyValues._get_n_evaluations` | `shapley_value.py:875-881` |
