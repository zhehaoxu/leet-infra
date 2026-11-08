# notes：Captum Integrated Gradients 求积与归因账

> 对应题面 `description.md`。这里放官方源码逐段解剖、实测数据、以及可复用的经验。

---

## 一、官方源码解剖

### 1.1 `captum/attr/_utils/approximation_methods.py`

```python
class Riemann(Enum):
    left = 1
    right = 2
    middle = 3
    trapezoid = 4

SUPPORTED_RIEMANN_METHODS = [
    "riemann_left", "riemann_right", "riemann_middle", "riemann_trapezoid",
]
SUPPORTED_METHODS: List[str] = SUPPORTED_RIEMANN_METHODS + ["gausslegendre"]

def approximation_parameters(method):
    if method in SUPPORTED_RIEMANN_METHODS:
        return riemann_builders(method=Riemann[method.split("_")[-1]])
    if method == "gausslegendre":
        return gauss_legendre_builders()
    raise ValueError("Invalid integral approximation method name: {}".format(method))
```

`method.split("_")[-1]` 取最后一段去 `Riemann` 枚举里查表 —— 所以
`riemann_trapezoid` → `Riemann.trapezoid`。

### 1.2 `riemann_builders` 的两个闭包（本题 §1.2 的出处）

```python
def step_sizes(n: int) -> List[float]:
    assert n > 1, "The number of steps has to be larger than one"
    deltas = [1 / n] * n
    if method == Riemann.trapezoid:
        deltas[0] /= 2
        deltas[-1] /= 2
    return deltas

def alphas(n: int) -> List[float]:
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

**关键观察（本题口径的由来）**：`torch.linspace(start, end, n)` 的步长是
`(end - start) / (n - 1)`。代入四个变体：

| 变体 | start | end | 步长 |
|---|---|---|---|
| left | `0` | `1-1/n` | `(1-1/n)/(n-1) = 1/n` |
| right | `1/n` | `1` | `(1-1/n)/(n-1) = 1/n` |
| middle | `1/(2n)` | `1-1/(2n)` | `(1-1/n)/(n-1) = 1/n` |
| trapezoid | `0` | `1` | `1/(n-1)` |

于是四组 `alphas` 的精确有理式是：

```
left      i/n
right     (i+1)/n
middle    (2i+1)/(2n)
trapezoid i/(n-1)
```

**四组全是精确有理数** —— 这正是本题能用「精确有理口径」逐位判定的原因。

### 1.3 `gauss_legendre_builders`

```python
def step_sizes(n: int) -> List[float]:
    assert n > 0, "The number of steps has to be larger than zero"
    return list(0.5 * np.polynomial.legendre.leggauss(n)[1])

def alphas(n: int) -> List[float]:
    assert n > 0, "The number of steps has to be larger than zero"
    return list(0.5 * (1 + np.polynomial.legendre.leggauss(n)[0]))
```

`leggauss(n)` 返回 `(nodes, weights)`，节点升序、权重与节点一一对应。

- `step_sizes = 0.5 * weights`：**求积权重**，不是等宽步长。
- `alphas = 0.5 * (1 + nodes)`：**节点**从 `[-1,1]` 缩放到 `[0,1]`。

两者都是 `n > 0`（不是 Riemann 的 `n > 1`）。

### 1.4 `captum/attr/_core/integrated_gradients.py::IntegratedGradients._attribute`

```python
def _attribute(self, inputs, baselines, target, additional_forward_args,
               n_steps=50, method="gausslegendre", step_sizes_and_alphas=None):
    if step_sizes_and_alphas is None:
        step_sizes_func, alphas_func = approximation_parameters(method)
        step_sizes, alphas = step_sizes_func(n_steps), alphas_func(n_steps)
    else:
        step_sizes, alphas = step_sizes_and_alphas

    # scaled_features' dim -> (bsz * #steps x inputs[0].shape[1:], ...)
    scaled_features_tpl = tuple(
        torch.cat(
            [baseline + alpha * (input - baseline) for alpha in alphas], dim=0
        ).requires_grad_()
        for input, baseline in zip(inputs, baselines)
    )
    ...
    grads = self.gradient_func(forward_fn=self.forward_func,
                               inputs=scaled_features_tpl, target_ind=expanded_target, ...)

    scaled_grads = [
        grad.contiguous().view(n_steps, -1)
        * torch.tensor(step_sizes).float().view(n_steps, 1).to(grad.device)
        for grad in grads
    ]

    total_grads = tuple(
        _reshape_and_sum(scaled_grad, n_steps, grad.shape[0] // n_steps, grad.shape[1:])
        for (scaled_grad, grad) in zip(scaled_grads, grads)
    )

    if not self.multiplies_by_inputs:
        attributions = total_grads
    else:
        attributions = tuple(
            total_grad * (input - baseline)
            for total_grad, input, baseline in zip(total_grads, inputs, baselines)
        )
    return attributions
```

要点：

1. **`torch.cat(..., dim=0)` 是 step-major**。每个 alpha 产生 `(B, F)` 的一段，
   拼起来是 `(S*B, F)`，第 `s*B + b` 行属于第 `s` 个 alpha 的第 `b` 个样本。
2. **`grad.contiguous().view(n_steps, -1)`** 把 `(S*B, F)` 压成 `(S, B*F)`，
   乘 `step_sizes.view(S, 1)` 时按行广播 —— 第 `s` 行的所有元素都乘
   `step_sizes[s]`。
3. **`torch.tensor(step_sizes).float()`** 把步长降到 float32 再乘。本题不追
   这个尾差。
4. **官方默认 `method="gausslegendre"`、`n_steps=50`**。

### 1.5 `captum/attr/_utils/common.py::_reshape_and_sum`

```python
def _reshape_and_sum(tensor_input: Tensor, num_steps: int,
                      num_examples: int, layer_size: Tuple[int, ...]) -> Tensor:
    tensor_input = tensor_input.reshape((num_steps, num_examples) + layer_size)
    return torch.sum(tensor_input, dim=0)
```

`reshape((S, B) + (F,))` 后 `sum(dim=0)` —— 沿 **step** 维求和。

### 1.6 `captum/attr/_utils/attribution.py::Attribution.compute_convergence_delta`

```python
start_out_sum = _sum_rows(start_out_eval)
end_out_sum   = _sum_rows(end_out_eval)
row_sums = [_sum_rows(attribution) for attribution in attributions]
attr_sum = torch.stack([cast(Tensor, sum(row_sum)) for row_sum in zip(*row_sums)])
_delta = attr_sum - (end_out_sum - start_out_sum)
```

`captum/attr/_utils/common.py::_sum_rows`：

```python
def _sum_rows(input: Tensor) -> Tensor:
    return input.reshape(input.shape[0], -1).sum(1)
```

`zip(*row_sums)` 里 `row_sums` 是「每个输入张量一个 `(B,)` 行和向量」的列表，
`zip` 后迭代的是**样本维**，所以 `attr_sum[b]` = 各输入张量在样本 `b` 上的行和
之**和**。

---

## 二、实测数据

### 2.1 `torch.linspace` 的 float32 尾差

```
torch.linspace(0, 1, 4) = [0.0, 0.3333333432674408, 0.6666666269302368, 1.0]
dtype: torch.float32
```

精确值应为 `[0, 1/3, 2/3, 1]`。第 2、3 项的偏差量级 ~1e-8 —— 这就是本题
把 Riemann 分支钉成「精确有理数」的原因。

### 2.2 五种方法的 `n = 4` 参数

```
left        deltas=[0.25, 0.25, 0.25, 0.25]           sum(d)=1.0
left        alphas=[0.0, 0.25, 0.5, 0.75]
right       deltas=[0.25, 0.25, 0.25, 0.25]           sum(d)=1.0
right       alphas=[0.25, 0.5, 0.75, 1.0]
middle      deltas=[0.25, 0.25, 0.25, 0.25]           sum(d)=1.0
middle      alphas=[0.125, 0.375, 0.625, 0.875]
trapezoid   deltas=[0.125, 0.25, 0.25, 0.125]         sum(d)=0.75   ← quirk
trapezoid   alphas=[0.0, 0.3333333432674408, 0.6666666269302368, 1.0]
gausslegendre deltas=[0.17392742256872673, 0.32607257743127327,
                      0.32607257743127327, 0.17392742256872673]  sum(d)=1.0
gausslegendre alphas=[0.06943184420297371, 0.33000947820757187,
                      0.6699905217924281,  0.9305681557970262]
```

注意 Gauss-Legendre 的 `deltas` **明显不是等宽**（0.1739 / 0.3261 交替），
一眼就能和 Riemann 区分开。

### 2.3 纯 Python Newton 法复现 numpy `leggauss`

Legendre 多项式递推 `P_k = ((2k-1)x·P_{k-1} - (k-1)P_{k-2})/k`，
导数 `P'_n = n(x·P_n - P_{n-1})/(x²-1)`，
权重 `w = 2/((1-x²)·P'_n²)`，
初值取 Chebyshev 型 `x = cos(π(i+0.75)/(n+0.5))`：

```
n= 1 max|dx|=0.000e+00 max|dw|=0.000e+00
n= 2 max|dx|=0.000e+00 max|dw|=2.220e-16
n= 3 max|dx|=0.000e+00 max|dw|=2.220e-16
n= 4 max|dx|=5.551e-17 max|dw|=4.441e-16
n= 6 max|dx|=1.110e-16 max|dw|=1.110e-16
n= 8 max|dx|=1.110e-16 max|dw|=3.469e-16
n=12 max|dx|=2.776e-17 max|dw|=5.551e-16
n=20 max|dx|=1.110e-16 max|dw|=9.298e-16
n=50 max|dx|=1.110e-16 max|dw|=2.670e-15
```

`n = 50` 时节点差仍只有 1.11e-16 —— 纯 Python 足以替代 numpy。

> 注意：Newton 迭代必须**按 x 升序重排**后再与官方对齐（官方 `leggauss`
> 返回升序节点）。初值 `cos(π(i+0.75)/(n+0.5))` 本身是降序的，直接用会
> 全表反序。

### 2.4 Gauss-Legendre 的结构性质

- 权重和 = 2（`leggauss` 定义），缩放后 `step_sizes` 之和 = 1。
- 节点和 = 0，因此 `alphas[i] + alphas[n-1-i] = 1`（关于 0.5 对称）。
- `n = 1`：node = 0，weight = 2 → `step_sizes = [1.0]`，`alphas = [0.5]`。
- `n = 2`：node = ±1/√3，weight = [1, 1] → `step_sizes = [0.5, 0.5]`，
  `alphas = [0.5 ∓ 1/(2√3)]`。

---

## 三、验证方式（三层）

### L1 语义层：与官方数学定义逐位一致

Riemann 分支用 `torch.linspace(dtype=torch.float64)` 重算官方 linspace 的
数学定义（`start + i*(end-start)/(n-1)`），与自写解的 `Fraction → float`
比较，要求偏差 ≤ 1e-15（实测逐位相等）。

Gauss-Legendre 分支与官方 `approximation_parameters("gausslegendre")`
直接调用 numpy `leggauss` 的结果比较，要求 ≤ 1e-12。

### L2 尾差层：定位官方 float32 实现的偏差

对 Riemann 分支，逐项比较自写解与官方 `torch.linspace`（float32）的输出。
每个分量的偏差必须 ≤ 1e-7（float32 量级）；若某项偏差超出，则**定位到具体
下标**报错。这保证「我们的精确值」与「官方的 float32 值」只差尾差，不差语义。

### L3 聚合层：官方张量逻辑对拍

照抄官方 `scaled_features` / `view(S,-1)` / `_reshape_and_sum` /
`multiply_by_inputs` 的张量写法（float64 版）与自写解对拍，偏差 ≤ 1e-12；
完备性侧照抄 `compute_convergence_delta` 的 `attr_sum` / `delta` 符号顺序对拍。

**对拍规模**：求积参数 4 变体 × 9 个 `n` + Gauss-Legendre 12 个 `n` +
聚合 160 组（各含 `multiply_by_inputs` 两种取值）+ 完备性 120 组 = **328 组**。

---

## 四、判题设计要点

### 4.1 为什么 Riemann 用「精确有理」而不是「跟随官方 float32」

官方 `torch.linspace` 走 float32，尾差 ~1e-8。若按官方值逐位判定，
解法必须**复刻 float32 的舍入**，这既不自然也不稳定；若放宽到 1e-6 容差，
又会掩盖真实错误（比如 `middle` 的 alpha 写成 `i/n` 的偏差是 `1/(2n)`，
`n = 50` 时只有 0.01 —— 仍能抓住，但 `trapezoid` 的 `i/n` vs `i/(n-1)` 在
大 `n` 时差 ~1/n² 就会被放过）。

精确有理口径同时解决三件事：
1. `trapezoid` 的 `i/(n-1)` vs `i/n` 在任何 `n` 下都判得出来；
2. `middle` 的 `(2i+1)/(2n)` vs `i/n` 判得出来；
3. 步长和 `1-1/n` vs `1` 判得出来。

### 4.2 为什么 Gauss-Legendre 用容差而不是精确

Legendre 节点是无理数，**没有**精确有理表示。这一支只能容差（1e-12）。
好在 Gauss-Legendre 分支只出现在 mode 一（求积参数），且它的语义差异
（权重 vs 等宽步长、节点 vs 等分点）都是**量级级别的**，容差 1e-12 完全
足够把它们和任何等价性错误区分开。

### 4.3 为什么 mode 二/三/四 不收 Gauss-Legendre

mode 二的 `scaled_features` 与 mode 三的 `attributions` 若混入无理数 alpha，
整个账就不再有唯一解（同一批浮点输入可以对应多个「正确」结果）。把
Gauss-Legendre 隔离在 mode 一，mode 二/三/四 保持「精确有理 → 逐位判定」，
判题强度最大。

### 4.4 锚点与网格的分工

- **手算锚点**：卡住官方 quirk 与符号约定（步长和 `1-1/n`、四种端点、
  step-major 布局、`delta` 符号、`multiply_by_inputs` 开关）。
- **网格对拍**：卡住组合爆炸（`n × B × F` 的交互）。
- **非零 delta 样例**：卡住 `attr_sum - (end - start)` 的符号（反号在
  delta = 0 的样例上照样通过，必须专门造非零样例）。

---

## 五、可复用经验（10 条）

1. **上游是浮点且有并列时，逐点 oracle 不成立**（841 的教训在本题的复现）。
   但本题更进一步：Riemann 的浮点误差**完全来自 `torch.linspace` 的
   float32**，而不是算法本身。把浮点回退到精确有理式（本题全部是
   `i/n` 形式），就能既避开尾差又不丢语义。
2. **先判断上游数学对象是否有理数**。有理 → 精确口径逐位判定；
   无理 → 容差判定，且要把无理分支**隔离在独立 mode**，不要让它污染
   其它 mode 的唯一解。
3. **`torch.linspace` 默认 float32**，任何以它为源的题都要显式声明精度
   口径，否则会出现「官方值与数学值差 1e-8」的对拍困难。
4. **官方 quirk 要单独手算锚点**。`trapezoid` 步长首尾减半导致和为 `1-1/n`，
   这是最容易写错又最容易被网格放过的点。
5. **`torch.cat(..., dim=0)` 的多段拼接是 step-major**（逐 alpha 段），
   不是 batch-major。这类「多段拼接」的布局是高频考点。
6. **`reshape((S,B)+shape).sum(dim=0)` 沿 step 维求和**。凡是
   `view(n_steps, -1)` 配套的 reduce，轴几乎总是 step 维。
7. **`zip(*row_sums)` 沿样本维 zip**，多输入张量的行和是「按样本求和」，
   不是「按张量求和」。
8. **符号顺序要单独用非零样例卡**。完备性残差写成 `a - (b - c)` 还是
   `(b - c) - a`，在「恰好完备」的样例上完全无法区分。
9. **纯 Python 可以替代 numpy 的 `leggauss`**：Legendre 递推 + Newton
   迭代，`n = 50` 时节点差 1.11e-16。**记得按节点升序重排**（Newton 初值
   是降序的）。这让 solution 可以保持零第三方依赖（库内纪律：
   `grep -rl "^import numpy" problems/pytorch/*/solution.py` 命中 0 个）。
10. **断言边界也可能是考点**：Riemann 是 `n > 1`，Gauss-Legendre 是 `n > 0`。
    同一个库里两套断言，读源码时容易只记一套。
