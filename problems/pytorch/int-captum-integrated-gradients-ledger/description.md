# Captum Integrated Gradients 的求积与归因账

> 官方回源：`pytorch/captum`（main）
> `captum/attr/_utils/approximation_methods.py` · `captum/attr/_core/integrated_gradients.py` · `captum/attr/_utils/common.py` · `captum/attr/_utils/attribution.py`

Integrated Gradients（IG）把「某个输入特征对最终输出的贡献」写成一条从
baseline 到 input 的路径积分：

```
F(input) - F(baseline) = ∫₀¹ (∇F(baseline + α·(input - baseline))) · (input - baseline) dα
```

官方用**有限步求积**近似这个积分：用一组步长 `step_sizes` 和一组路径参数
`alphas` 采样若干点，在每个点求一次梯度，加权求和得到 `total_grads`，
再乘上 `(input - baseline)` 得到 `attributions`。

本题把这条链路拆成四段**纯数值账**，逐段重建官方行为。不做前向传播、
不做自动微分、不做 GPU kernel —— 只算「参数怎么来、布局怎么排、加权怎么加、
delta 怎么减」。

---

## 一、mode 一：`quadrature_params`

> 入参：`method`（字符串）、`n_steps`（整数）
> 返回：`{"step_sizes": [...], "alphas": [...]}`

### 1.1 官方有两种求积方法族

官方 `approximation_parameters(method)` 只认两种：

```python
SUPPORTED_RIEMANN_METHODS = ["riemann_left", "riemann_right",
                             "riemann_middle", "riemann_trapezoid"]
SUPPORTED_METHODS = SUPPORTED_RIEMANN_METHODS + ["gausslegendre"]
```

传入其它名字必须抛 `ValueError`。

### 1.2 Riemann 四变体

官方 `riemann_builders` 的两个闭包：

```python
def step_sizes(n):
    assert n > 1, "The number of steps has to be larger than one"
    deltas = [1 / n] * n
    if method == Riemann.trapezoid:
        deltas[0] /= 2
        deltas[-1] /= 2
    return deltas

def alphas(n):
    assert n > 1, "The number of steps has to be larger than one"
    if method == Riemann.trapezoid:
        return torch.linspace(0, 1, n).tolist()
    elif method == Riemann.left:
        return torch.linspace(0, 1 - 1 / n, n).tolist()
    elif method == Riemann.middle:
        return torch.linspace(1 / (2 * n), 1 - 1 / (2 * n), n).tolist()
    elif method == Riemann.right:
        return torch.linspace(1 / n, 1, n).tolist()
```

**四个变体的 `alphas` 端点各不相同**，这是本题最容易被忽略的一组事实：

| method | 官方 linspace | `alphas[0]` | `alphas[n-1]` | 含 0 | 含 1 |
|---|---|---|---|---|---|
| `riemann_left` | `linspace(0, 1-1/n, n)` | `0` | `(n-1)/n` | ✅ | ❌ |
| `riemann_right` | `linspace(1/n, 1, n)` | `1/n` | `1` | ❌ | ✅ |
| `riemann_middle` | `linspace(1/(2n), 1-1/(2n), n)` | `1/(2n)` | `1-1/(2n)` | ❌ | ❌ |
| `riemann_trapezoid` | `linspace(0, 1, n)` | `0` | `1` | ✅ | ✅ |

把 `torch.linspace` 的数学定义展开（步长 `(end-start)/(n-1)`），这四组
`alphas` 全都是**精确有理数**：

```
left      alphas[i] = i/n          (i = 0..n-1)
right     alphas[i] = (i+1)/n
middle    alphas[i] = (2i+1)/(2n)
trapezoid alphas[i] = i/(n-1)
```

`step_sizes` 同理：

```
left / right / middle :  step_sizes = [1/n] * n
trapezoid             :  step_sizes = [1/(2n)] + [1/n]*(n-2) + [1/(2n)]
```

> ⚠️ **官方 quirk**：只有 `trapezoid` 的步长首尾减半，因此它的
> `step_sizes` 之和是 `1 - 1/n`，**不是 1**。其余三个变体的步长和为 1。

### 1.3 Gauss-Legendre

官方 `gauss_legendre_builders` 走的是**完全不同的语义**：

```python
def step_sizes(n):
    assert n > 0, "The number of steps has to be larger than zero"
    return list(0.5 * np.polynomial.legendre.leggauss(n)[1])   # ← 求积权重

def alphas(n):
    assert n > 0, "The number of steps has to be larger than zero"
    return list(0.5 * (1 + np.polynomial.legendre.leggauss(n)[0]))  # ← 节点
```

两个关键差异：

1. **`step_sizes` 不是等宽步长，而是 Legendre 求积权重**（缩放到区间长度 1）。
   权重之和恰好为 1，但每个权重互不相等。
2. **`alphas` 不是等分点，而是求积节点**从 `[-1, 1]` 缩放到 `[0, 1]`：
   `alphas[i] = 0.5 * (1 + node[i])`，节点按升序排列。

另外，断言边界也变了：Riemann 要求 `n > 1`，Gauss-Legendre 要求 `n > 0`
（`n = 1` 合法）。

### 1.4 本题口径（重要）

官方 `torch.linspace` 默认走 **float32**，会引入 ~1e-8 量级的尾差
（例：`riemann_trapezoid` 在 `n = 4` 时第二个 alpha，float32 值是
`0.3333333432674408`，精确值是 `1/3 = 0.3333333333333333`）。

为了让账本可判定，本题把口径钉死为：

- **Riemann 四变体**：`step_sizes` 与 `alphas` 一律按**精确有理数**计算
  （用 `fractions.Fraction`），输出时转 `float`（Fraction → float 是正确舍入，
  唯一确定）。因此这些结果可以**逐位相等**判定。
- **Gauss-Legendre**：节点是 Legendre 多项式的无理数根，无法用有理数表达，
  用**相对容差 1e-12** 判定。求解方式不限（纯 Python Newton 法即可，
  与 numpy `leggauss` 实测 `n = 50` 时节点差 ≤ 1.11e-16、权重差 ≤ 2.67e-15）。

### 1.5 非法输入

| 情形 | 行为 |
|---|---|
| `method` 不在 `SUPPORTED_METHODS` 里 | 抛 `ValueError` |
| Riemann 变体且 `n_steps <= 1` | 抛 `ValueError` |
| `gausslegendre` 且 `n_steps <= 0` | 抛 `ValueError` |

---

## 二、mode 二：`scaled_features`

> 入参：`input`（`[B][F]`）、`baseline`（`[B][F]`）、`method`、`n_steps`
> 返回：`{"scaled_features": [S*B][F]}`

官方 `IntegratedGradients._attribute` 的第一步：

```python
scaled_features_tpl = tuple(
    torch.cat(
        [baseline + alpha * (input - baseline) for alpha in alphas], dim=0
    ).requires_grad_()
    for input, baseline in zip(inputs, baselines)
)
```

即：对每个 alpha 算一整批 `baseline + alpha * (input - baseline)`，
再沿 `dim=0` 拼起来。得到的形状是 `(S*B, F)`。

> ⚠️ **布局是 step-major，不是 batch-major**。
> 第 `s*B + b` 行属于**第 `s` 个 alpha**下的**第 `b` 个样本**。
>
> 例：`B = 2, S = 2`，则
> ```
> 行 0 = alpha0 × 样本0        行 1 = alpha0 × 样本1
> 行 2 = alpha1 × 样本0        行 3 = alpha1 × 样本1
> ```
> 若错排成 batch-major（行 0 = alpha0×样本0、行 1 = alpha1×样本0、……），
> 后续加权聚合会全盘错位。

只有 Riemann 四变体参与本 mode 的判定（Gauss-Legendre 的 alpha 是无理数，
与 mode 一的容差口径混用会让账本失去唯一解）。本 mode 全程按精确有理数计算。

---

## 三、mode 三：`ig_aggregate`

> 入参：`input`（`[B][F]`）、`baseline`（`[B][F]`）、`grads`（`[S*B][F]`）、
> `method`、`n_steps`、`multiply_by_inputs`（默认 `True`）
> 返回：`{"total_grads": [B][F], "attributions": [B][F]}`

`grads` 按**官方 flatten 后的布局**给出：长度 `S*B`，第 `s*B + b` 行是
第 `s` 个 alpha 下第 `b` 个样本的梯度（与 mode 二的 `scaled_features` 同序）。

官方聚合段：

```python
scaled_grads = [
    grad.contiguous().view(n_steps, -1)
    * torch.tensor(step_sizes).float().view(n_steps, 1).to(grad.device)
    for grad in grads
]

total_grads = tuple(
    _reshape_and_sum(scaled_grad, n_steps, grad.shape[0] // n_steps, grad.shape[1:])
    for (scaled_grad, grad) in zip(scaled_grads, grads)
)

if self.multiplies_by_inputs:
    attributions = tuple(
        total_grad * (input - baseline)
        for total_grad, input, baseline in zip(total_grads, inputs, baselines)
    )
else:
    attributions = total_grads
```

其中官方 `_reshape_and_sum` 是：

```python
def _reshape_and_sum(tensor_input, num_steps, num_examples, layer_size):
    tensor_input = tensor_input.reshape((num_steps, num_examples) + layer_size)
    return torch.sum(tensor_input, dim=0)
```

### 三步账

1. **加权**：第 `s` 段的每一行梯度都乘上 `step_sizes[s]`。
2. **沿 step 维求和**：`total_grads[b][f] = Σ_s scaled_grads[s*B + b][f]`。
   注意求和沿的是 **step** 维（`dim=0`），不是 batch 维。
3. **乘输入差**：`multiply_by_inputs=True` 时
   `attributions[b][f] = total_grads[b][f] * (input[b][f] - baseline[b][f])`；
   为 `False` 时 `attributions == total_grads`。

> ⚠️ 官方 `torch.tensor(step_sizes).float()` 把步长**降到 float32** 再乘。
> 本题钉死精确有理数口径，不追这个尾差（与 mode 一同）。

只有 Riemann 四变体参与本 mode 的判定。

---

## 四、mode 四：`ig_completeness`

> 入参：`attributions`（`[B][F]`）、`start_out`（`[B][K]`）、`end_out`（`[B][K]`）
> 返回：`{"attr_sum": [B], "start_out_sum": [B], "end_out_sum": [B], "delta": [B]}`

IG 的**完备性公理**（completeness axiom）说：归因之和应当等于函数在终点与
起点的输出差。官方 `Attribution.compute_convergence_delta` 把它算成一个残差：

```python
row_sums = [_sum_rows(attribution) for attribution in attributions]
attr_sum = torch.stack([cast(Tensor, sum(row_sum)) for row_sum in zip(*row_sums)])
_delta = attr_sum - (end_out_sum - start_out_sum)
```

其中官方 `_sum_rows` 是 `input.reshape(input.shape[0], -1).sum(1)`，即把每个
样本的行拉平求和。

单输入张量时：

```
attr_sum[b]      = Σ_f attributions[b][f]
end_out_sum[b]   = Σ_k end_out[b][k]
start_out_sum[b] = Σ_k start_out[b][k]
delta[b]         = attr_sum[b] - (end_out_sum[b] - start_out_sum[b])
```

> ⚠️ **符号顺序**：`attr_sum` 在前，`end_out_sum - start_out_sum` 在后。
> 写反了 delta 整体取反，完备性成立的样例（delta = 0）仍然会通过，
> 所以必须用 delta ≠ 0 的样例来卡这个点。

> ⚠️ `zip(*row_sums)` 沿 **sample** 维 zip：多个输入张量时，
> `attr_sum[b]` 是各张量第 `b` 样本行和的**总和**。本题只给一个输入张量，
> 因此就是该样本全部特征的归因之和。

本 mode 与 `method` 无关，全程精确有理数计算。

---

## 五、易错点清单

| # | 易错点 | 正确做法 |
|---|---|---|
| 1 | 以为 `trapezoid` 的步长和是 1 | 是 `1 - 1/n`（首尾减半） |
| 2 | 四个变体的 `alphas` 端点写成一样 | 逐变体按 §1.2 表 |
| 3 | `middle` 的 alpha 写成 `i/n` | 是 `(2i+1)/(2n)` |
| 4 | `trapezoid` 的 alpha 写成 `i/n` | 是 `i/(n-1)` |
| 5 | Gauss-Legendre 的 `step_sizes` 写成等宽 `1/n` | 是 Legendre 求积权重的一半 |
| 6 | Gauss-Legendre 的 `alphas` 写成等分点 | 是缩放后的求积节点 |
| 7 | Gauss-Legendre 也断言 `n > 1` | 是 `n > 0` |
| 8 | `scaled_features` 排成 batch-major | 是 step-major |
| 9 | `grads` 按 batch-major 解释 | 是 step-major（第 `s*B+b` 行） |
| 10 | 沿 batch 维求和 | 沿 step 维求和 |
| 11 | 忽略 `multiply_by_inputs` 开关 | 为 `False` 时 `attributions == total_grads` |
| 12 | delta 写成 `(end - start) - attr_sum` | 是 `attr_sum - (end - start)` |
| 13 | 以为 delta = 0 就是完备性 | 完备时 delta = 0，但反号也 = 0，要用非零样例卡 |
| 14 | 用 float32 linspace 的值 | 本题 Riemann 分支用精确有理数 |
