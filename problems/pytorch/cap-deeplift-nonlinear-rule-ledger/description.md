# Captum DeepLIFT 非线性传播规则账

## 背景

DeepLIFT（Deep LIFT）是一种基于「乘性归因」的归因方法：它不直接算梯度，而是对网络里的
每个非线性层追问一句——**这一层的输出，相对于 baseline（参考输入）改变了多少？**
然后把这个改变量按比例摊回到输入上。

摊回的方式就是「传播规则」（propagation rule）。Captum 在
`captum/attr/_core/deep_lift.py` 里给每类非线性层配了一条规则函数：

| 层 | 规则函数 |
|---|---|
| `nn.ReLU` / `nn.ELU` / `nn.LeakyReLU` / `nn.Sigmoid` / `nn.Tanh` / `nn.Softplus` | `nonlinear` |
| `nn.Softmax` | `softmax` |
| `nn.MaxPool1d` / `nn.MaxPool2d` / `nn.MaxPool3d` | `maxpool` |

本题要你用纯 Python 复现这三条规则（外加它们共同依赖的差分函数 `_compute_diffs`）的
**精确数值口径**。这些规则里有好几处「看起来合理但官方不是这么写」的细节，
本题的判题会把它们逐个钉死。

## 口径约定

所有 mode 的入参都是**展平的一维列表**，不涉及任何张量形状推断：

- `inputs` / `outputs`：长度必须为**偶数**。官方用 `chunk(2)` 沿第 0 维把它们拆成
  **前半 = 实际输入（actual）**、**后半 = 参考输入（reference）**。
- `grad_input` / `grad_output`：梯度，形状与对应的 `inputs` / `outputs` 一致。
- 所有出参统一为 `float`（内部能走精确有理数的部分走 `fractions.Fraction`）。
- 非法输入一律抛 `ValueError`。

---

## mode 1：`chunk_diffs`

复现官方 `_compute_diffs`：

```python
def _compute_diffs(inputs, outputs):
    input, input_ref = inputs.chunk(2)
    output, output_ref = outputs.chunk(2)
    delta_in = input - input_ref
    delta_out = output - output_ref
    return torch.cat(2 * [delta_in]), torch.cat(2 * [delta_out])
```

**签名**

```python
chunk_diffs(inputs, outputs)
    → {"delta_in": [...], "delta_out": [...], "half": int}
```

**要点**

1. 差分方向是 `actual - reference`（不是反过来）。
2. ★ 返回值是 `torch.cat(2 * [delta])` —— 每份差分被**复制成两份**，
   所以返回长度等于**入参长度**，不是入参的一半。
3. 返回的两半内容完全相同。
4. `half` = `len(inputs) // 2`。

---

## mode 2：`nonlinear_rule`

复现官方 `nonlinear`：

```python
def nonlinear(module, inputs, outputs, grad_input, grad_output, eps=1e-10):
    delta_in, delta_out = _compute_diffs(inputs, outputs)
    new_grad_inp = torch.where(
        abs(delta_in) < eps, grad_input, grad_output * delta_out / delta_in
    )
    return new_grad_inp
```

**签名**

```python
nonlinear_rule(inputs, outputs, grad_input, grad_output, eps=1e-10)
    → {"delta_in", "delta_out", "new_grad_in", "used_fallback", "n_fallback"}
```

**要点**

1. ★ 判据是**严格小于**：`abs(delta_in) < eps`。当 `abs(delta_in)` **恰好等于** `eps` 时，
   走的是**除法支**。
2. 除法支的算式是 `grad_output * delta_out / delta_in`（先乘后除）。
3. `delta_in` 为负时，结果符号随之为负。
4. `delta_in` 为 0（或 `abs(delta_in) < eps`）时走 `grad_input` 支。
5. `grad_input` / `grad_output` 的长度必须与 `delta_in` 相同 —— 也就是与 `inputs` 同长
   （因为 `_compute_diffs` 把差分复制成了两份）。
6. `eps` 的**默认值是 `1e-10`**（不传 `eps` 时用它）。
7. `used_fallback` 是逐元素的布尔列表，标记该位置是否走了 `grad_input` 支；
   `n_fallback` 是其中 `True` 的个数。

---

## mode 3：`softmax_rule`

复现官方 `softmax`：

```python
def softmax(module, inputs, outputs, grad_input, grad_output, eps=1e-10):
    delta_in, delta_out = _compute_diffs(inputs, outputs)
    grad_input_unnorm = torch.where(
        abs(delta_in) < eps, grad_input, grad_output * delta_out / delta_in
    )
    n = grad_input.numel()
    new_grad_inp = grad_input_unnorm - grad_input_unnorm.sum() * 1 / n
    return new_grad_inp
```

**签名**

```python
softmax_rule(inputs, outputs, grad_input, grad_output, eps=1e-10)
    → {"unnorm", "total", "n", "shift", "new_grad_in"}
```

**要点**

1. 先算未归一化版本 `unnorm`（与 `nonlinear_rule` 的 `new_grad_in` 完全一样）。
2. ★ `n = grad_input.numel()` —— 是**全部元素个数**，不是 batch 数，
   也不是差分的「单侧长度」。由于 `_compute_diffs` 复制了两份，
   `n` 实际上等于 `len(inputs)`。
3. ★ 减去的是 `unnorm.sum() * 1 / n` —— 一个**全局标量**，
   **不是**逐行（逐 half）减均值。
4. 修正后所有元素之和为 0。
5. `total` = `unnorm` 的元素和；`shift` = 被减掉的那个标量（`total / n`）。

---

## mode 4：`maxpool_rule`

复现官方 `maxpool`（以 `MaxPool1d` 为例；`2d` / `3d` 只是多几个空间轴，
在本 mode 的一维口径下语义相同）：

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

        _, indices = pool_func(module.input, module.kernel_size, module.stride,
                               module.padding, module.dilation, module.ceil_mode, True)

        grad_output_updated = grad_output
        unpool_grad_out_delta, unpool_grad_out_ref_delta = torch.chunk(
            unpool_func(
                grad_output_updated * delta_out,
                indices,
                module.kernel_size, module.stride, module.padding,
                list(module.input.shape),
            ),
            2,
        )

    unpool_grad_out_delta = unpool_grad_out_delta + unpool_grad_out_ref_delta
    unpool_grad_out_delta = torch.cat(2 * [unpool_grad_out_delta])

    new_grad_inp = torch.where(
        abs(delta_in) < eps, grad_input[0], unpool_grad_out_delta / delta_in
    )
    return new_grad_inp
```

**签名**

```python
maxpool_rule(inputs, outputs, grad_input, grad_output, indices, eps=1e-10)
    → {"delta_in", "delta_out_xmax", "delta_out", "scaled",
       "unpool_delta", "unpool_ref", "merged", "new_grad_in", "used_fallback"}
```

**口径**

设 `n_feat = len(inputs) // 2`（单侧输入长度），`n_out = len(outputs) // 2`（单侧输出长度）：

| 量 | 长度 | 含义 |
|---|---|---|
| `inputs` | `2 * n_feat` | 前半 = actual 输入，后半 = reference 输入 |
| `outputs` | `2 * n_out` | 前半 = actual 输出，后半 = reference 输出 |
| `grad_input` | `2 * n_feat` | 与 `inputs` 同形 |
| `grad_output` | `2 * n_out` | 与 `outputs` 同形 |
| `indices` | `2 * n_out` | 前半 = actual 侧的窗口命中位置，后半 = reference 侧的窗口命中位置 |

**要点**

1. ★ `delta_out_xmax = torch.max(output, output_ref)` —— **逐元素**取 actual 与
   reference 两半的最大值。
2. ★ `delta_out = cat([xmax - output_ref, output - xmax])`：
   - 前半（`n_out` 个）= `xmax - output_ref`，即「xmax 相对 reference 的超出量」，
     **恒非负**；
   - 后半（`n_out` 个）= `output - xmax`，即「actual 输出相对 xmax 的缺口」，
     **恒非正**；当 reference 侧更大时它为负。
3. ★ `grad_output` 与 `outputs` **同形**（长度 `2 * n_out`），
   `grad_output * delta_out` 是**逐元素**相乘，不是广播。
4. `scaled` = 上面那个逐元素乘积（长度 `2 * n_out`）。
5. **unpool 分配**：`indices` 给出每个输出窗口的梯度该放到哪个输入位置。
   - 前半用 `indices[:n_out]`（actual 侧的窗口命中位置）配 `scaled[:n_out]`；
   - 后半用 `indices[n_out:]`（reference 侧）配 `scaled[n_out:]`；
   - 分别写进 `unpool_delta`（长 `n_feat`）与 `unpool_ref`（长 `n_feat`）。
   - ★ **同一个位置被多个窗口命中时是「后写覆盖」，不是累加**
     （这是 `torch.max_unpool` 的实际行为）。
6. `merged` = `unpool_delta + unpool_ref`（逐元素，长 `n_feat`），
   然后**复制成两份**（长 `2 * n_feat`）。
7. ★ eps 分支取的是 `grad_input[0]` —— 沿第 0 维的**第一半**
   （不是 `grad_input` 的第 0 个元素）。因此在 eps 分支下，
   **两半都用第一半的 `grad_input` 值**。
8. 其余位置结果是 `merged / delta_in`。

---

## 常见错误清单

1. `chunk(2)` 拆成两半后，差分只算一份、不复制 → 返回长度错。
2. `delta_in` 方向写成 `reference - actual`。
3. `chunk_diffs` 里把 `half` 当成「单侧长度」，但返回时只给单侧。
4. eps 判据写成 `<=`（官方是严格 `<`）。
5. eps 判据写成 `abs(delta_in) < eps` 但把 `eps` 默认值记成 `1e-6`（官方是 `1e-10`）。
6. `softmax` 的 `n` 用「单侧长度」而不是 `grad_input.numel()`。
7. `softmax` 做成逐行（逐 half）减均值，而不是减一个全局标量。
8. `softmax` 忘了先算 `unnorm` 就归一化。
9. `maxpool` 的 `grad_output` 当成「单侧长度」，靠广播相乘。
10. `maxpool` 的 `xmax` 取 actual 侧（不取两半逐元素最大值）。
11. `maxpool` 的 `delta_out` 两半写反（写成 `output - output_ref` 与它的相反）。
12. `maxpool` 的 unpool 用**累加**而不是**后写覆盖**。
13. `maxpool` 的 `grad_input[0]` 取成「第 0 个元素」而不是「第一半」。
14. `maxpool` 的 `merged` 只算一半就返回（忘了复制成两份）。
15. `maxpool` 的 `unpool_delta` / `unpool_ref` 用了错误的 `indices` 半区。

## 自检

```bash
python starter.py            # 骨架自检
```

提交前请确认：

- 四个 mode 的返回键名与题面完全一致。
- 所有出参是 `float`（不是 `Fraction`）。
- 非法输入抛 `ValueError`。
