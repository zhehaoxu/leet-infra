# Captum DeepLIFT 源码解剖：非线性传播规则账

> 官方仓库：`pytorch/captum`（GitHub repo id `204734444`，默认分支 `main`）
> 官方文件：`captum/attr/_core/deep_lift.py`
> 本题关注的四个函数：`_compute_diffs` / `nonlinear` / `softmax` / `maxpool`

---

## 1. 这套规则在 DeepLIFT 里的位置

`DeepLift` 的工作方式是：在模型的每个非线性层上挂 forward hook 与 backward hook。
forward hook 记下这一层的 `input` 与 `output`；backward hook 触发时，
拿 `grad_input` / `grad_output` 调该层对应的**传播规则**，把规则算出来的
「乘性因子」替换掉原本的梯度。

规则函数通过一张查表绑定：

```python
SUPPORTED_NON_LINEAR = {
    nn.ReLU: nonlinear,
    nn.ELU: nonlinear,
    nn.LeakyReLU: nonlinear,
    nn.Sigmoid: nonlinear,
    nn.Tanh: nonlinear,
    nn.Softplus: nonlinear,
    nn.MaxPool1d: maxpool1d,
    nn.MaxPool2d: maxpool2d,
    nn.MaxPool3d: maxpool3d,
    nn.Softmax: softmax,
}
```

`maxpool1d` / `2d` / `3d` 三个入口只是把不同的 `pool_func` / `unpool_func`
（`F.max_pool1d` 与 `F.max_unpool1d` 等）传给同一个 `maxpool` 主体：

```python
def maxpool1d(module, inputs, outputs, grad_input, grad_output, eps=1e-10):
    return maxpool(
        module, F.max_pool1d, F.max_unpool1d,
        inputs, outputs, grad_input, grad_output, eps=eps,
    )
```

所以本题的 `maxpool_rule` 只需要实现 `maxpool` 主体那一套语义。

### 1.1 `module.input` 与 `inputs` 是同一个张量

这一点决定了 `indices` 是怎么来的。`_forward_pre_hook` 里：

```python
def _forward_pre_hook(self, module, inputs):
    inputs = _format_tensor_into_tuples(inputs)
    module.input = inputs[0].clone().detach()
```

而 `_backward_hook` 里传给规则的是：

```python
multipliers = SUPPORTED_NON_LINEAR[type(module)](
    module,
    module.input,
    module.output,
    grad_input,
    grad_output,
    eps=self.eps,
)
```

⇒ `maxpool` 里的 `module.input` 就是 `inputs`（同一个张量）。
所以 `pool_func(module.input, ...)` 是在**拼接后**的输入上算 `indices` 的，
而 `unpool` 的 `output_size = list(module.input.shape)` 也取的是这个拼接后的形状。

> ★ 这是本题最容易踩空的一处：如果 `module.input` 填成「单侧」的输入，
> `indices` 的形状会与 `grad_output * delta_out` 不匹配，官方直接报错。

---

## 2. `_compute_diffs`：差分的「复制两份」语义

```python
def _compute_diffs(inputs: Tensor, outputs: Tensor) -> Tuple[Tensor, Tensor]:
    input, input_ref = inputs.chunk(2)
    output, output_ref = outputs.chunk(2)
    delta_in = input - input_ref
    delta_out = output - output_ref

    return torch.cat(2 * [delta_in]), torch.cat(2 * [delta_out])
```

### ★ 要点 2.1：`cat(2 * [delta])` 让长度翻倍

`torch.cat(2 * [delta_in])` 等价于 `torch.cat([delta_in, delta_in])`，
即把同一份差分**首尾相接拼成两份**。

所以：

- `delta_in` 的长度 == `inputs` 的长度（**不是**一半）；
- 前一半与后一半内容完全相同；
- 这一点直接决定了 `nonlinear` 里 `grad_input` / `grad_output`
  的长度必须与 `inputs` **同长**（而不是与单侧长度同长）。

实测（官方函数直接调用）：

```python
inputs  = torch.tensor([1., 2., 3., 4., 0.5, 1.5, 2.5, 3.5])
outputs = torch.tensor([10., 20., 30., 40., 5., 15., 25., 35.])
di, do = _compute_diffs(inputs, outputs)
# di.shape == (8,)  do.shape == (8,)
# di.tolist() == [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
```

### ★ 要点 2.2：差分方向是 `actual - reference`

`chunk(2)` 沿 dim 0 拆：**前半是 actual，后半是 reference**。
于是 `input - input_ref` = actual 减 reference。

方向反了的话，整条链的符号都会反。

---

## 3. `nonlinear`：where 的两个分支

```python
def nonlinear(module, inputs, outputs, grad_input, grad_output, eps=1e-10):
    delta_in, delta_out = _compute_diffs(inputs, outputs)

    new_grad_inp = torch.where(
        abs(delta_in) < eps, grad_input, grad_output * delta_out / delta_in
    )
    return new_grad_inp
```

### ★ 要点 3.1：判据是**严格小于**

`torch.where(cond, a, b)` 在 `cond` 为真时取 `a`。
条件写的是 `abs(delta_in) < eps` —— **严格小于**。

实测（`delta_in` 恰为 `0.5`）：

| `eps` | `abs(0.5) < eps` | 走哪支 |
|---|---|---|
| `0.5` | `0.5 < 0.5` → False | **除法支** |
| `0.5000001` | `0.5 < 0.5000001` → True | `grad_input` 支 |
| `0.4999999` | `0.5 < 0.4999999` → False | 除法支 |

⇒ 判题里必须有一个「恰好等于 eps」的用例，否则 `<` 与 `<=` 区分不出来。

### ★ 要点 3.2：除法支是 `grad_output * delta_out / delta_in`

先乘后除。浮点上 `a * b / c` 与 `a / c * b` 一般差一个 ulp，
但本题的参考解走 `Fraction` 精确有理数，两者**完全相等** ⇒
这是一条**等价变异**，不能拿来当判分点。

### ★ 要点 3.3：`delta_in` 为负时结果符号随之为负

实测：`delta_in = -2`、`delta_out = 5` → 结果 `-2.5`。
没有取绝对值，也没有 `clamp`。

### ★ 要点 3.4：`grad_input` / `grad_output` 长度必须等于 `delta_in`

由于 `_compute_diffs` 复制了两份，`delta_in` 长度 = `inputs` 长度。
若 `grad_input` 长度不匹配，官方直接报：

```
RuntimeError: The size of tensor a (4) must match the size of tensor b (8)
```

### ★ 要点 3.5：默认 `eps = 1e-10`

官方签名里写死了 `eps: float = 1e-10`。
判题里要有一个**不传 `eps`** 的用例把默认值钉住 ——
用一个 `|delta_in|` 落在 `1e-10` 与 `1e-6` 之间的输入，
「默认 1e-10」与「默认 1e-6」两种实现就会给出不同结果。

---

## 4. `softmax`：全局归一化修正

```python
def softmax(module, inputs, outputs, grad_input, grad_output, eps=1e-10):
    delta_in, delta_out = _compute_diffs(inputs, outputs)

    grad_input_unnorm = torch.where(
        abs(delta_in) < eps, grad_input, grad_output * delta_out / delta_in
    )
    # normalizing
    n = grad_input.numel()

    # updating only the first half
    new_grad_inp = grad_input_unnorm - grad_input_unnorm.sum() * 1 / n
    return new_grad_inp
```

### ★ 要点 4.1：`n = grad_input.numel()` 是**全部元素个数**

- 不是 batch 数；
- 不是差分的「单侧长度」；
- 由于差分被复制成两份，`n` 实际等于 `len(inputs)`。

实测（`grad_input` 长 8）：`n == 8`。

### ★ 要点 4.2：减的是**一个全局标量**

`grad_input_unnorm.sum() * 1 / n` 是一个 0 维张量，
广播到 `grad_input_unnorm` 的每个元素上。

⇒ 这**不是**逐行（逐 batch / 逐 half）减均值。

实测：`unnorm = [10]*4 + [20]*4` → `sum = 120`，`n = 8` → 减 `15` →
`[-5]*4 + [5]*4`。

> ⚠️ **判题设计的坑**：如果 `unnorm` 的两半恰好对称（前半和 = 后半和），
> 「全局减标量」与「逐 half 减均值」会给出**相同**结果 ⇒ 该用例失去区分能力。
> 所以判题必须用**不对称**的数据，并显式断言两种候选不同。

### ★ 要点 4.3：修正后总和为 0

这是全局标量修正的直接推论，可以作为一条便宜的性质断言
（但注意：逐 half 减均值也会让总和为 0，所以这条**单独用不足以区分**）。

### ★ 要点 4.4：注释 `updating only the first half` 与代码不符

官方源码里有一行注释写 `# updating only the first half`，
但代码实际上是 `grad_input_unnorm - grad_input_unnorm.sum() * 1 / n`，
**对所有元素做了修正**。这行注释是从更早的实现遗留下来的误导。

⇒ 本题以**代码**为准，不要照注释实现。

---

## 5. `maxpool`：cross-maximum 与 unpool 分配

```python
def maxpool(module, pool_func, unpool_func, inputs, outputs,
            grad_input, grad_output, eps=1e-10):
    with torch.no_grad():
        input, input_ref = inputs.chunk(2)
        output, output_ref = outputs.chunk(2)

        delta_in = input - input_ref
        delta_in = torch.cat(2 * [delta_in])

        delta_out_xmax = torch.max(output, output_ref)
        delta_out = torch.cat([delta_out_xmax - output_ref, output - delta_out_xmax])

        _, indices = pool_func(
            module.input, module.kernel_size, module.stride,
            module.padding, module.dilation, module.ceil_mode, True,
        )
        grad_output_updated = grad_output
        unpool_grad_out_delta, unpool_grad_out_ref_delta = torch.chunk(
            unpool_func(
                grad_output_updated * delta_out,
                indices,
                module.kernel_size, module.stride, module.padding,
                list(cast(torch.Size, module.input.shape)),
            ),
            2,
        )

    unpool_grad_out_delta = unpool_grad_out_delta + unpool_grad_out_ref_delta
    unpool_grad_out_delta = torch.cat(2 * [unpool_grad_out_delta])

    if grad_input.shape != inputs.shape:
        raise AssertionError(...)

    new_grad_inp = torch.where(
        abs(delta_in) < eps, grad_input[0], unpool_grad_out_delta / delta_in
    )
    return new_grad_inp
```

### ★ 要点 5.1：`delta_out_xmax` 是**逐元素**的最大值

`torch.max(output, output_ref)` 是张量逐元素的最大值（不是 `torch.max` 的
「降维取最大」用法）。

实测（`output = [2, 4]`、`output_ref = [6, 8]`）：`xmax = [6, 8]`。

### ★ 要点 5.2：`delta_out` 两半的语义完全不同

```python
delta_out = torch.cat([delta_out_xmax - output_ref, output - delta_out_xmax])
```

- 前半 = `xmax - output_ref` —— 「xmax 相对 reference 超出多少」，**恒非负**；
- 后半 = `output - xmax` —— 「actual 相对 xmax 差多少」，**恒非正**；
  当 reference 侧更大时为负。

这正是「cross-maximum」设计的意图：**梯度归给输出更大的那一侧**。
actual 更大时，增量（正）落在前半；reference 更大时，缺口（负）落在后半。

实测（`output = [2, 4]`、`output_ref = [6, 8]`）：
`delta_out = [0, 0, -4, -4]`。

### ★ 要点 5.3：`grad_output` 与 `outputs` **同形**

官方 backward hook 传进来的 `grad_output` 形状与该层输出一致，
而 `outputs` 已经是**拼接后**的（actual + reference），所以
`grad_output` 长度 = `2 * n_out`。

于是 `grad_output_updated * delta_out` 是**逐元素**相乘，长度 `2 * n_out`。

> ⚠️ **常见错误**：把 `grad_output` 当成「单侧长度」，靠广播相乘。
> 实测：`grad_output` 形状 `(1, 1, n_out)` 时，
> `prod` 会广播成 `(2, 1, n_out)` —— 数值上能跑，
> 但语义是错的（两半共用同一份梯度），判题必须排除。

### ★ 要点 5.4：unpool 的 scatter 是**后写覆盖**

`unpool_func` 在官方里是 `torch.nn.functional.max_unpool1d`。
它把每个输出窗口的梯度写到 `indices` 指定的输入位置。
**多个窗口命中同一位置时，后写的覆盖先写的**，而不是累加。

实测（`indices = [1, 1, 2]`、`prod = [10, 20, 30]`）：
结果 `[0, 20, 30, 0, 0]` —— 位置 1 拿到的是 **20**（第二个窗口的值），不是 `10+20=30`。

把 `prod` 反过来（`[30, 20, 10]`）→ 结果 `[0, 20, 10, 0, 0]`，
位置 1 仍是 **20**（后写的那份）⇒ 确认是「后写覆盖」，与数值大小无关。

### ★ 要点 5.5：两半 scatter 结果先相加，再复制两份

```python
unpool_grad_out_delta = unpool_grad_out_delta + unpool_grad_out_ref_delta
unpool_grad_out_delta = torch.cat(2 * [unpool_grad_out_delta])
```

`torch.chunk(unpool(...), 2)` 把 unpool 结果（形状与 `module.input` 相同，
即拼接后的形状）沿 dim 0 拆成 actual 半与 reference 半；
两者**逐元素相加**，再 `cat(2 * [...])` **复制成两份**。

⇒ 最终 `merged` 长度 = `2 * n_feat`，两半相同。

### ★ 要点 5.6：eps 分支取 `grad_input[0]`（第一半）

`grad_input[0]` 是沿 **dim 0** 的第 0 个切片，形状 `(1, ..., n_feat)`，
**不是** `grad_input` 的第 0 个元素（那是个标量）。

实测（`grad_input = [[7,8,9,10],[11,12,13,14]]`）：
`grad_input[0]` = `[7,8,9,10]`（形状 `(1,4)`）。

由于 `torch.where` 会广播，eps 分支下**两半都用第一半的值**。

实测（`delta_in` 全 0、`grad_input = [7,8,9,10,11,12,13,14]`）：
输出 `[7,8,9,10, 7,8,9,10]` —— 后半也是 `7,8,9,10`，不是 `11,12,13,14`。

### ★ 要点 5.7：`grad_input` 形状必须等于 `inputs`

官方有一道显式断言：

```python
if grad_input.shape != inputs.shape:
    raise AssertionError("A problem occurred during maxpool modul's backward pass. ...")
```

即 `grad_input` 与 `inputs` **同形**（都是拼接后的形状）。

---

## 6. 官方源码的三个复现坑

### 坑 1：`maxpool` 依赖 torch 的 pooling 实现

`maxpool` 主体本身是纯 tensor 运算，但它把 `pool_func` / `unpool_func`
当参数收进来（`F.max_pool1d` / `F.max_unpool1d` 等）。
要复现它，就必须接受 torch pooling 的两个外部约束：

**(a) `indices` 由 pooling 在 `module.input` 上算出**

- `pool_func(module.input, kernel_size, stride, padding, dilation, ceil_mode, True)`
  —— 第三个位置参数是 `return_indices=True`；
- `indices` 形状 = `module.input` 去掉空间维后的形状，
  即**拼接后**的 `(2, 1, n_out)`；
- `max_pool1d` 的 tie-break 是**首个最大值**（实测 `[5,5,1,1]` k=2 s=2 → `indices=[0,2]`）。

**(b) `torch.max_unpool1d` 对 `output_size` 有合法性校验**

官方传的是 `list(module.input.shape)`，而 torch 会检查它与
「由 `indices` 与 kernel/stride 反推出来的自然输出大小」是否相容：

```
ValueError: invalid output_size "[6]" (dim 0 must be between 4 and 6)
```

⇒ 必须满足 `M == floor((L - k) / s) + 1`（`L` = 单侧输入长度，
`M` = 单侧输出长度）。这不是 captum 的限制，是 torch 的；
但对拍时必须遵守，否则官方函数直接抛异常。

### 坑 2：`module.input` 必须是拼接后的张量

如果 `module.input` 填成单侧输入：

- `pool_func` 算出的 `indices` 只有一半的 batch 维，
  与 `grad_output * delta_out` 的 `(2, 1, n_out)` 不匹配 ⇒ 官方报
  `Expected shape of indices to be same as that of the input tensor`；
- `unpool` 的 `output_size` 也会取错。

### 坑 3：`grad_output` 的形状会「碰巧」跑通

`grad_output` 传单侧形状时，`grad_output * delta_out` 会广播成
`(2, 1, n_out)`，官方函数**不报错**、还能给出数值。
但那是错的语义（两半共用同一份梯度）。
⇒ 判题必须用「`grad_output` 与 `outputs` 同形」的口径，
并显式断言长度，否则这个错误实现会静悄悄存活。

---

## 7. 三层验证结构

本题采用与官方源码直接对拍的三层验证：

| 层 | 内容 | 作用 |
|---|---|---|
| 官方层 | `captum/attr/_core/deep_lift.py` 用 AST 切片**逐字抽取**的 `_compute_diffs` / `nonlinear` / `softmax` / `maxpool` | 语义的唯一真相 |
| 参考解层 | 纯 Python + `Fraction` 实现 | 提交者要写的东西 |
| oracle 层 | 不依赖参考解、按官方算式独立重写一遍（纯 float） | 防止参考解与 oracle「同源出错」 |

对拍结果：

```
WORST official-vs-ref = 5.551e-17
WORST oracle-vs-ref   = 5.551e-17
```

（`5.551e-17` 是 softmax 归一化时的浮点舍入，量级 ~1e-17 属正常。）

官方层用 `ast.parse` + `ast.get_source_segment` 抽取源码，
所以**跑的就是 upstream 的原文**，不是手抄的副本。

---

## 8. mutation 的三条设计原则（本题再次验证）

### 原则 1：等价变异要主动剔除

- `go * delta_out / delta_in` → `go / delta_in * delta_out`：
  `Fraction` 精确有理数下**完全相等** ⇒ 等价变异，剔除。
- `delta_out[0] + delta_out[1]` → `delta_out[1] + delta_out[0]`：
  两半内容相同 ⇒ 等价变异，剔除。

判分点必须落在**真正会变**的地方。

### 原则 2：不可观测的差异要设计专门的用例

- 「eps 严格小于 vs 小于等于」：必须有一个 `|delta_in|` **恰好等于** `eps` 的用例。
- 「全局减标量 vs 逐 half 减均值」：必须用**不对称**数据，
  并显式断言两种候选不同，否则该用例没有区分能力。
- 「默认 eps 是 1e-10 还是 1e-6」：必须有一个**不传 `eps`**、
  且 `|delta_in|` 落在两者之间的用例。

### 原则 3：同一语义不要在多层重复校验

`logq` 式的「多处都乘 mask」是 846 的教训。本题里
`delta_in` / `delta_out` 被复制成两份这件事，在 `chunk_diffs`、
`nonlinear`、`softmax`、`maxpool` 四个 mode 里都会出现。
判题只在**每一层各测一次**该层特有的东西，
不把「长度翻倍」在四个 check 里各测一遍 ——
那样一旦长度语义错了，只会看到同一个错误重复报四遍，浪费判分信息。

---

## 9. 可复用的经验

1. **`torch.cat(2 * [x])` 是长度翻倍，不是「不变」** —— 读官方代码时
   看到 `cat(2 * [...])` 要立刻在脑中标出「输出长度 = 2 × 输入长度」。
2. **`torch.where` 的三参数里，条件是严格还是非严格比较要看源码**，
   差一个 `=` 就是一条独立的判分点。
3. **`grad_input.numel()` 是元素总数**，`sum()` 是**全局**标量 ——
   归一化类操作先确认是「全局」还是「按行」。
4. **`grad_input[0]` 与 `grad_input[0][0]` 完全不同**：
   前者是切片（保留形状），后者是标量。
5. **官方注释可能过时**（`softmax` 里那句 `# updating only the first half`
   与代码不符）—— 以代码为准。
6. **「能跑通」不等于「语义对」**：官方 `maxpool` 在 `grad_output` 形状错误时
   照样广播出正确形状并给出数值。这类「静悄悄的错误」必须靠
   判题里显式断言口径来排除。
7. **对拍要分层**：官方 / 参考解 / 独立 oracle 三层都收敛，
   才能排除「参考解和 oracle 同源出错」。
8. **反着用例验证**：把已知错误实现注入参考解，确认判题能抓住 ——
   比只看「参考解全过」可靠得多。

---

## 10. 官方源码位置索引

| 内容 | 位置 |
|---|---|
| `SUPPORTED_NON_LINEAR` 映射表 | `captum/attr/_core/deep_lift.py` 末尾 |
| `_compute_diffs` | 同上，靠近文件尾部 |
| `nonlinear` | 同上 |
| `softmax` | 同上 |
| `maxpool` | 同上 |
| `maxpool1d` / `maxpool2d` / `maxpool3d` | 同上 |
| `_forward_pre_hook`（设 `module.input`） | `DeepLift` 类内 |
| `_backward_hook`（调规则） | `DeepLift` 类内 |
