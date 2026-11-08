# order 843 · 源码解剖：Captum LRP 的 z-rule 传播账

> 官方源：`pytorch/captum`（main 分支，仓库 id `204734444`）
> 回源文件：`captum/attr/_core/lrp.py`（482 行）、`captum/attr/_utils/lrp_rules.py`（218 行）、
> `captum/attr/_utils/custom_modules.py`（`Addition_Module`）

---

## 一、官方 LRP 的完整执行流程

`LRP.attribute` 的骨架（`lrp.py`）：

```python
self._original_state_dict = self.model.state_dict()
self.layers = []
self._get_layers(self.model)          # 收集叶子层
self._check_and_attach_rules()         # 给每层挂 rule

# 1. 第一次前向：改权重 + 存激活
output = self._compute_output_and_change_weights(inputs, target, additional_forward_args)

# 2. 第二次前向 + 反向
self._register_forward_hooks()
normalized_relevances = self.gradient_func(self._forward_fn_wrapper, inputs, target, ...)
relevances = tuple(
    normalized_relevance * output.reshape((-1,) + (1,) * (normalized_relevance.dim() - 1))
    for normalized_relevance in normalized_relevances
)
```

**关键**：`output` 是**第一次前向**（原权重）的输出；`normalized_relevances` 是**第二次前向**
（改写权重）的反向结果。两者乘在一起得到最终 attribution。

### 1.1 `_get_layers` —— 只收叶子层

```python
for layer in model.children():
    if len(list(layer.children())) == 0:
        self.layers.append(layer)
    else:
        self._get_layers(layer)
```

**⚠️ 官方陷阱**：`nn.Linear` 直接当 model 时 `model.children()` 是**空的** → `self.layers` 为空
→ **一条 rule 都挂不上** → attribution 退化成「纯梯度 × output」。实测三种 rule 给出的 attribution
**完全相同**（`0.1255614012479782, -0.08742005378007889, ...`），因为根本没进 rule 逻辑。

必须用 `nn.Sequential(Linear, Linear, ...)` 包一层，`layers` 才非空。
本题口径因此定为「只建模叶子线性层组成的栈」，等价于 `nn.Sequential`。

### 1.2 `_compute_output_and_change_weights` —— 第一次前向

```python
self._register_weight_hooks()          # 注册 forward_hook_weights
output = _run_forward(self.model, inputs, target, additional_forward_args)
finally:
    self._remove_forward_hooks()
self._register_pre_hooks()              # 注册 forward_pre_hook_activations
```

`forward_hook_weights`（在 `lrp_rules.py`）：

```python
module.activations[device] = tuple(input.data for input in inputs)   # 存【输入】激活
self._manipulate_weights(module, inputs, outputs)                   # 改写权重
```

注意 forward hook 是**前向之后**才执行的，所以第一次前向本身用的是**原权重**，
存下来的激活也是原权重下的激活。

### 1.3 `forward_pre_hook_activations` —— 还原激活

```python
for input, activation in zip(inputs, module.activations[device]):
    input.data = activation
```

第二次前向时，每层的输入被**强制还原**成第一次前向存下的激活。

**这直接决定了「第二次前向 = 改写权重 × 原激活」**，而不是「改写权重链式前向」。

### 1.4 `forward_hook` —— 注册 backward hook

```python
def forward_hook(self, module, inputs, outputs):
    inputs = _format_tensor_into_tuples(inputs)
    self._has_single_input = len(inputs) == 1
    self._handle_input_hooks = []
    for input in inputs:
        if not hasattr(input, "hook_registered"):
            input_hook = self._create_backward_hook_input(input.data)
            self._handle_input_hooks.append(input.register_hook(input_hook))
            input.hook_registered = True
    output_hook = self._create_backward_hook_output(outputs.data)
    self._handle_output_hook = outputs.register_hook(output_hook)
    return outputs.clone()
```

**注意 `outputs.clone()`**：官方返回克隆张量，所以 backward 时梯度先流过 clone 再到 `outputs`，
`outputs` 上挂的 hook 才拿得到正确的 grad。

---

## 二、z-rule 的两个 hook

### 2.1 output hook（`lrp_rules.py`）

```python
def _create_backward_hook_output(self, outputs):
    def _backward_hook_output(grad):
        sign = torch.sign(outputs)
        sign[sign == 0] = 1                      # ← 官方 quirk：sign(0) = +1
        relevance = grad / (outputs + sign * self.STABILITY_FACTOR)
        self.relevance_output[grad.device] = grad.data
        return relevance
    return _backward_hook_output
```

- `STABILITY_FACTOR = 1e-9`（基类常量）；`EpsilonRule.__init__(epsilon=1e-9)` 会把它覆盖成实例属性。
- **`sign[sign == 0] = 1`**：输出恰好为 0 时分母是 `+eps`。
  若漏掉这条，遇到 `outputs == 0` 会直接 `ZeroDivisionError`（本题的 mutation
  `sign-zero-not-plus-one` 就是这样被抓到的）。
- `relevance_output` 存的是**原始 grad**（不是 relevance 本身），所以 `IdentityRule` 拿它当
  relevance 用是官方的一个语义 bug。

### 2.2 input hook

```python
def _create_backward_hook_input(self, inputs):
    def _backward_hook_input(grad):
        relevance = grad * inputs
        ...
        grad.replace_out = relevance
        return relevance
    return _backward_hook_input
```

`inputs` 是**第一次前向**的激活（被 pre_hook 还原过），所以 `rel_in` 乘的是原激活。

### 2.3 反向链

PyTorch 的 autograd 会按 `outputs → inputs` 的顺序触发 hook，于是：

```
最顶层：grad = e_target
  rel_out[L] = grad / (outputs[L] + sign*eps)
  （autograd 继续把 rel_out[L] 当作 grad 往回传）
  grad_input[L] = rel_out[L] @ W'[L]
  → input hook 触发：rel_in[L] = grad_input[L] * activations[L-1]
  → 这个返回值替换了继续往回传的 grad
下一层：upstream = rel_in[L]
  rel_out[L-1] = upstream / (outputs[L-1] + sign*eps)
  ...
```

**⚠️ 关键点：`upstream` 逐层传递的是上一层的 `rel_in`（已经乘过原激活）**，
不是 `rel_out`。本题实现里如果每层都重置成 one-hot，跨层传播就断了
（mutation `stack-no-upstream-propagation`）。

---

## 三、三条规则的权重改写

```python
class EpsilonRule(PropagationRule):
    def __init__(self, epsilon: float = 1e-9) -> None:
        self.STABILITY_FACTOR = epsilon
    def _manipulate_weights(self, module, inputs, outputs) -> None:
        pass                                     # ← 权重不动

class GammaRule(PropagationRule):
    def __init__(self, gamma: float = 0.25, set_bias_to_zero: bool = False) -> None:
        self.gamma = gamma
        self.set_bias_to_zero = set_bias_to_zero
    def _manipulate_weights(self, module, inputs, outputs) -> None:
        if hasattr(module, "weight"):
            module.weight.data = (
                module.weight.data + self.gamma * module.weight.data.clamp(min=0)
            )
        if self.set_bias_to_zero and hasattr(module, "bias"):
            if module.bias is not None:
                module.bias.data = torch.zeros_like(module.bias.data)

class Alpha1_Beta0_Rule(PropagationRule):
    def _manipulate_weights(self, module, inputs, outputs) -> None:
        if hasattr(module, "weight"):
            module.weight.data = module.weight.data.clamp(min=0)
        if self.set_bias_to_zero and hasattr(module, "bias"):
            if module.bias is not None:
                module.bias.data = torch.zeros_like(module.bias.data)
```

要点：

1. **`GammaRule` 的 quirk**：`W + gamma * clamp(W, min=0)` —— 只有**正**权重被放大 `(1+gamma)`，
   负权重加上 `gamma * 0` 即保持不变。**不是**「全体乘 `(1+gamma)`」。
2. **`Alpha1_Beta0_Rule`**：`clamp(min=0)` 把负权重清零（Deep-Taylor）。
3. **`EpsilonRule` 的 `_manipulate_weights` 是 `pass`** —— 只改 `STABILITY_FACTOR`，权重偏置都不动。
4. **`set_bias_to_zero`** 只有 `GammaRule` / `Alpha1_Beta0_Rule` 有这个构造参数；
   `EpsilonRule.__init__` 只收 `epsilon`。
5. 官方还检查 `hasattr(module, "weight")` / `module.bias is not None` —— 本题所有层都带偏置，不涉及。

### `IdentityRule` 是官方 bug

```python
class IdentityRule(EpsilonRule):
    def _create_backward_hook_input(self, inputs):
        def _backward_hook_input(grad):
            return self.relevance_output[grad.device]      # ← KeyError!
        return _backward_hook_input
```

`relevance_output` 是在 output hook 里填的，而 input hook 在 backward 中**先于** output hook 触发
（对同一层而言），所以 `self.relevance_output[device]` 抛 `KeyError: device(type='cpu')`。
实测单层与两层都必崩。**本题不收 `identity` 规则**，非法 rule 一律 `ValueError`。

---

## 四、归一化与 delta

### 4.1 归一化

```python
relevances = tuple(
    normalized_relevance * output.reshape((-1,) + (1,) * (normalized_relevance.dim() - 1))
    for normalized_relevance in normalized_relevances
)
```

`normalized_relevance` 是 `rel_in`（最输入层），`output` 是**原权重**下的 target 值。
所以：

```
attribution = rel_in[0] * output_target          （output_target 来自原权重链式前向）
```

**注意**：LRP 不是「归一化成 sum=1 再乘 output」，而是「relevance 本身就已经是归一化过的
（因为分母 `outputs + sign*eps` ≈ outputs，量级约 1/特征值），再乘 output」。
所以 `sum(attr)` 一般**不等于** `output`。

### 4.2 delta（完备性）

```python
def compute_convergence_delta(self, attributions, output):
    if isinstance(attributions, tuple):
        for attr in attributions:
            summed_attr = cast(Tensor, sum(_sum_rows(attr) for attr in attributions))
    else:
        summed_attr = _sum_rows(attributions)
    return output.flatten() - summed_attr.flatten()
```

- `_sum_rows(input) = input.reshape(input.shape[0], -1).sum(1)`（每行求和）。
- **`output` 在前** —— 与 Captum 的 Integrated Gradients
  （`captum/attr/_utils/attribution.py::compute_convergence_delta`，`attr_sum - (end_out - start_out)`）
  **符号相反**。这是 842 与 843 之间的重要划界点。
- 官方 docstring 明确：使用带 epsilon 的规则时「relevance is absorbed during propagation」，
  所以 delta 非零。

---

## 五、精确有理数口径的由来

官方是 float64 + `eps = 1e-9`（二进制浮点）。本题把 eps 钉成**精确有理数 `1/10^9`**：

- `1e-9` 在十进制里就是 `1/10^9`；官方 float64 的 `1e-9` 与精确有理差 ~6.2e-26 相对，
  被 float64 的舍入完全掩盖。
- 题目把 eps 钉成有理数，等于**把官方实现里被浮点掩盖的语义显式化**——
  与 841（精确余弦交叉相乘）、842（Riemann 分支钉精确有理）是同一套方法论。
- 全链路 `Fraction` 精确 → 唯一可能的误差是「精确有理 → float」的一次舍入（相对 ~1e-16），
  判题容差 `1e-12` 恰好覆盖它，而任何语义错误都会远超这个量级。

**官方对拍实测**（`/tmp/verify843.py`，captum 0.9.0 float64）：

| 用例 | attr 最大绝对误差 | 相对误差 |
|---|---|---|
| 单层 epsilon | 8.882e-16 | 1.110e-16 |
| 单层 gamma(1/4) | 8.882e-16 | 1.236e-16 |
| 单层 alpha1_beta0 | 4.441e-16 | 2.019e-17 |
| 单层 gamma + set_bias_to_zero | 1.110e-16 | 1.798e-17 |
| 两层 gamma/epsilon | 1.421e-14 | 2.572e-16 |
| 两层 epsilon/gamma | 7.105e-15 | 1.200e-16 |
| 两层 alpha1_beta0/gamma | 0 | 0 |
| 三层混合 | 8.882e-15 | 4.202e-16 |

**最大相对误差 4.2e-16**，全部是官方 float64 自身的尾差。

---

## 六、可复用的经验（10 条）

1. **官方包可以装，而且值得装**。`uv pip install --python .venv-eval/bin/python --target /tmp/captum_pkgs --no-deps captum`
   → captum 0.9.0；再逐个补 `matplotlib` → `pillow` → `pyparsing/cycler/fonttools/kiwisolver/python-dateutil/contourpy`。
   装在 `/tmp` 的 target 目录，**不污染判题环境**；对拍脚本用 `PYTHONPATH=/tmp/captum_pkgs` 运行。
   这是本题能拿到「官方真值」而不是「读代码推测」的关键。

2. **`diff` 安装版与 main 分支**：`lrp_rules.py` 差异只有类型注解；`lrp.py` 差异只有 typing/pyre 注释与
   `log_usage(part_of_slo=True)`。**语义完全一致**，可以放心用安装版做对拍。

3. **必须用 `nn.Sequential` 包一层**。`nn.Linear` 直接当 model 时 `model.children()` 为空 →
   `self.layers` 为空 → 一条 rule 都不挂 → 三种 rule 的 attribution 完全相同。
   这是本题最早踩的坑（探针 `probe843.py` 里三种 rule 输出一样）。

4. **`rule` 会在 `_restore_model` 里被 `del`**（`lrp.py::_remove_rules`），所以想在跑完后检查
   `layer.rule.relevance_output` 必须**插桩**（patch `_restore_model` 或 patch hook 工厂函数）。

5. **`GammaRule` 的 gamma 必须是 float**（`self.gamma * module.weight.data.clamp(min=0)` 是张量乘标量）。
   传字符串会报 `TypeError: only integer tensors of a single element can be converted to an index`。
   题目侧用精确有理数字符串 `"1/4"`，对拍脚本侧要 `float(Fraction(...))` 转一下。

6. **hook 捕获的 `outputs` 是「第二次前向」的值**，`inputs` 是「第一次前向」的激活。
   两者来源不同，这是最容易搞混的地方。

7. **反向链上传的是上一层的 `rel_in`**（不是 `rel_out`）。官方 hook 轨迹实测：
   `in` hook 的 `grad` = 上一层的 `rel_out @ W'`，它返回的 `relevance` = 该值 × 原激活，
   这个返回值就是下一层 `out` hook 收到的 `grad`。

8. **`sign(0) = +1` 必须专门造样例**。锚点 `W=[[1,1]], b=[-2], inputs=[1,1]` → `outputs=[0]`，
   分母 = `+eps`，`relevance_out = [1/eps] = [1e9]`，`attribution = [0, 0]`。
   漏掉这条会 `ZeroDivisionError`，mutation 一抓一个准。

9. **官方 delta 与 IG 的 delta 符号相反**。LRP：`output - attr_sum`（`lrp.py`）；
   IG：`attr_sum - (end_out - start_out)`（`attr/_utils/attribution.py`）。
   两题在同一合集里，务必在题面显式写明，否则解答者极易混用。

10. **实现里把 `upstream` 显式传参比在函数内重置更安全**。初版把 `one_hot(target)` 写在
    `_layer_ledger` 内部，导致每层都重置 → 单层对、多层错（官方对拍在「两层」用例上炸出
    `rel=0.219897`）。修法是把 `upstream` 变成参数，最顶层传 one-hot，其余层传上一层的 `rel_in`。
    **跨层语义必须靠「独立 oracle + 官方对拍」双验证，逐层打印 hook 轨迹最快定位。**

---

## 七、本题的三层验证结构

1. **官方层**：用安装版 captum 的 `LRP` 直接跑 float64，对拍 `attribution`（相对误差 ≤ 4.2e-16）。
2. **独立 oracle 层**：`test_cases.py` 里另写一份 `Fraction` 实现（与 solution 是**两条独立代码路径**），
   逐位比对 `activations` / `outputs` / `relevance_out` / `relevance_in` / `attribution`。
3. **变异层**：39 个单点变异全部 CAUGHT（覆盖规则改写、sign/eps、z-rule 分解、两次前向语义、
   层序、delta 符号、非法输入）。

**mutation 中抓到的真 bug（都是自己写的时候真会犯的）**：

| 变异 | 抓到的错误 |
|---|---|
| `stack-no-upstream-propagation` | 每层重置 upstream → 跨层传播断裂 |
| `stack-second-forward-chained` | 第二次前向写成改写权重链式 |
| `stack-attr-uses-rewritten-output` | 归一化基准用了改写后的 output |
| `stack-relevance-order-reversed` | 层序翻转（rel_in[0] 取到输出层） |
| `rewrite-gamma-scales-all` | gamma 误实现为全体放大 |
| `sign-zero-not-plus-one` | 漏掉 sign(0)=+1 |

**判题缺口（mutation 存活暴露出来的）**：`stack-empty-layers-allowed` 与
`stack-out-of-range-target-allowed` 一开始存活 —— 判题没覆盖「空层栈」和「target 越界」，
补上对应断言后全绿。**mutation 存活 = 判题缺口**，这条经验值得固化。
