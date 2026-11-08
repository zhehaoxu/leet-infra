# order 843 · Source dissection: Captum LRP's z-rule relevance ledger

> Official source: `pytorch/captum` (main branch, repo id `204734444`)
> Files fetched: `captum/attr/_core/lrp.py` (482 lines), `captum/attr/_utils/lrp_rules.py` (218 lines),
> `captum/attr/_utils/custom_modules.py` (`Addition_Module`)

---

## 1. The full execution flow of official LRP

The skeleton of `LRP.attribute` (`lrp.py`):

```python
self._original_state_dict = self.model.state_dict()
self.layers = []
self._get_layers(self.model)          # collect leaf layers
self._check_and_attach_rules()         # attach a rule to each layer

# 1. first forward: rewrite weights + stash activations
output = self._compute_output_and_change_weights(inputs, target, additional_forward_args)

# 2. second forward + backward
self._register_forward_hooks()
normalized_relevances = self.gradient_func(self._forward_fn_wrapper, inputs, target, ...)
relevances = tuple(
    normalized_relevance * output.reshape((-1,) + (1,) * (normalized_relevance.dim() - 1))
    for normalized_relevance in normalized_relevances
)
```

**Key point**: `output` comes from the **first forward** (original weights); `normalized_relevances`
comes from the **second forward** (rewritten weights). The final attribution is the product of the two.

### 1.1 `_get_layers` — leaf layers only

```python
for layer in model.children():
    if len(list(layer.children())) == 0:
        self.layers.append(layer)
    else:
        self._get_layers(layer)
```

**⚠️ Official trap**: when `nn.Linear` is used as the model directly, `model.children()` is **empty**
→ `self.layers` is empty → **not a single rule gets attached** → the attribution degenerates into
"plain gradient × output". Empirically all three rules produce **identical** attributions
(`0.1255614012479782, -0.08742005378007889, ...`) because the rule logic is never reached.

You must wrap the layers in `nn.Sequential(Linear, Linear, ...)` so that `layers` is non-empty.
This problem therefore models "a stack of leaf linear layers", which is equivalent to `nn.Sequential`.

### 1.2 `_compute_output_and_change_weights` — the first forward

```python
self._register_weight_hooks()          # register forward_hook_weights
output = _run_forward(self.model, inputs, target, additional_forward_args)
finally:
    self._remove_forward_hooks()
self._register_pre_hooks()              # register forward_pre_hook_activations
```

`forward_hook_weights` (in `lrp_rules.py`):

```python
module.activations[device] = tuple(input.data for input in inputs)   # stash the INPUT activation
self._manipulate_weights(module, inputs, outputs)                   # rewrite the weights
```

Note that a forward hook runs **after** the forward pass, so the first forward itself uses the
**original weights**, and the stashed activations are the activations under the original weights.

### 1.3 `forward_pre_hook_activations` — restoring activations

```python
for input, activation in zip(inputs, module.activations[device]):
    input.data = activation
```

During the second forward, the input of every layer is **forcibly restored** to the activation
stashed during the first forward.

**This is exactly what makes "the second forward = rewritten weights × original activations"
true, rather than "a chained forward with rewritten weights".**

### 1.4 `forward_hook` — registering backward hooks

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

**Note `outputs.clone()`**: the official implementation returns a cloned tensor, so during backward
the gradient first flows through the clone and only then reaches `outputs`, which is what allows
the hook attached to `outputs` to observe the correct grad.

---

## 2. The two hooks of the z-rule

### 2.1 output hook (`lrp_rules.py`)

```python
def _create_backward_hook_output(self, outputs):
    def _backward_hook_output(grad):
        sign = torch.sign(outputs)
        sign[sign == 0] = 1                      # ← official quirk: sign(0) = +1
        relevance = grad / (outputs + sign * self.STABILITY_FACTOR)
        self.relevance_output[grad.device] = grad.data
        return relevance
    return _backward_hook_output
```

- `STABILITY_FACTOR = 1e-9` (a base-class constant); `EpsilonRule.__init__(epsilon=1e-9)`
  overrides it as an instance attribute.
- **`sign[sign == 0] = 1`**: when the output is exactly 0 the denominator is `+eps`.
  If you drop this line, hitting `outputs == 0` raises `ZeroDivisionError` outright
  (this is exactly how the mutation `sign-zero-not-plus-one` was caught).
- `relevance_output` stores the **raw grad** (not the relevance itself), which is why
  `IdentityRule` using it as the relevance is an official semantic bug.

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

`inputs` is the activation from the **first forward** (restored by the pre-hook), so `rel_in`
multiplies the original activation.

### 2.3 The backward chain

PyTorch's autograd fires the hooks in the order `outputs → inputs`, so:

```
top layer: grad = e_target
  rel_out[L] = grad / (outputs[L] + sign*eps)
  (autograd keeps propagating rel_out[L] as the grad)
  grad_input[L] = rel_out[L] @ W'[L]
  → the input hook fires: rel_in[L] = grad_input[L] * activations[L-1]
  → this return value replaces the grad that keeps flowing back
next layer: upstream = rel_in[L]
  rel_out[L-1] = upstream / (outputs[L-1] + sign*eps)
  ...
```

**⚠️ Key point: the `upstream` propagated to the next layer is the previous layer's `rel_in`
(already multiplied by the original activation)**, not `rel_out`. If your implementation resets
the upstream to a one-hot vector inside every layer, the cross-layer propagation breaks
(mutation `stack-no-upstream-propagation`).

---

## 3. The weight rewrite of the three rules

```python
class EpsilonRule(PropagationRule):
    def __init__(self, epsilon: float = 1e-9) -> None:
        self.STABILITY_FACTOR = epsilon
    def _manipulate_weights(self, module, inputs, outputs) -> None:
        pass                                     # ← weights untouched

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

Takeaways:

1. **`GammaRule`'s quirk**: `W + gamma * clamp(W, min=0)` — only **positive** weights are scaled by
   `(1+gamma)`; negative weights get `gamma * 0` added and therefore stay unchanged.
   It is **not** "scale everything by `(1+gamma)`".
2. **`Alpha1_Beta0_Rule`**: `clamp(min=0)` zeroes the negative weights (Deep-Taylor).
3. **`EpsilonRule`'s `_manipulate_weights` is `pass`** — it only changes `STABILITY_FACTOR`,
   leaving weights and bias untouched.
4. **`set_bias_to_zero` exists only as a constructor argument of `GammaRule` / `Alpha1_Beta0_Rule`**;
   `EpsilonRule.__init__` only accepts `epsilon`.
5. The official code also checks `hasattr(module, "weight")` / `module.bias is not None` — every layer
   in this problem carries a bias, so that branch is not exercised here.

### `IdentityRule` is an official bug

```python
class IdentityRule(EpsilonRule):
    def _create_backward_hook_input(self, inputs):
        def _backward_hook_input(grad):
            return self.relevance_output[grad.device]      # ← KeyError!
        return _backward_hook_input
```

`relevance_output` is filled in by the output hook, while the input hook fires **before** the
output hook during backward (for a given layer), so `self.relevance_output[device]` raises
`KeyError: device(type='cpu')`. Empirically it crashes for both the one-layer and the two-layer
case. **This problem does not accept the `identity` rule**; any illegal rule raises `ValueError`.

---

## 4. Normalization and delta

### 4.1 Normalization

```python
relevances = tuple(
    normalized_relevance * output.reshape((-1,) + (1,) * (normalized_relevance.dim() - 1))
    for normalized_relevance in normalized_relevances
)
```

`normalized_relevance` is `rel_in` (of the input-most layer), and `output` is the **target value
under the original weights**. Hence:

```
attribution = rel_in[0] * output_target          (output_target comes from the original-weight chained forward)
```

**Note**: LRP is not "normalize to sum=1 and then multiply by output". Rather, "the relevance is
already normalized (because the denominator `outputs + sign*eps` ≈ outputs, magnitude roughly
1/eigenvalue), and then multiplied by output".
So `sum(attr)` generally **does not equal** `output`.

### 4.2 delta (completeness)

```python
def compute_convergence_delta(self, attributions, output):
    if isinstance(attributions, tuple):
        for attr in attributions:
            summed_attr = cast(Tensor, sum(_sum_rows(attr) for attr in attributions))
    else:
        summed_attr = _sum_rows(attributions)
    return output.flatten() - summed_attr.flatten()
```

- `_sum_rows(input) = input.reshape(input.shape[0], -1).sum(1)` (row-wise sum).
- **`output` comes first** — the **opposite sign** from Captum's Integrated Gradients
  (`captum/attr/_utils/attribution.py::compute_convergence_delta`, `attr_sum - (end_out - start_out)`).
  This is the important boundary between problem 842 and problem 843.
- The official docstring states explicitly: when using a rule with epsilon, "relevance is absorbed
  during propagation", so the delta is non-zero.

---

## 5. Where the exact-rational convention comes from

The official implementation is float64 with `eps = 1e-9` (binary floating point). This problem
pins eps to the **exact rational `1/10^9`**:

- `1e-9` is exactly `1/10^9` in decimal; the official float64 `1e-9` differs from the exact
  rational by ~6.2e-26 relatively, which is completely masked by float64 rounding.
- Pinning eps to a rational amounts to **making explicit the semantics that the official
  implementation hides behind floating point** — the same methodology as problem 841 (exact
  cross-multiplied cosine) and problem 842 (Riemann branch pinned to an exact rational).
- The whole chain runs exactly in `Fraction` → the only possible error is the single rounding of
  "exact rational → float" (relative ~1e-16), which the judging tolerance `1e-12` comfortably
  covers, while any semantic error is orders of magnitude larger.

**Measured against the official implementation** (`/tmp/verify843.py`, captum 0.9.0 float64):

| Case | max absolute error of attr | relative error |
|---|---|---|
| one layer epsilon | 8.882e-16 | 1.110e-16 |
| one layer gamma(1/4) | 8.882e-16 | 1.236e-16 |
| one layer alpha1_beta0 | 4.441e-16 | 2.019e-17 |
| one layer gamma + set_bias_to_zero | 1.110e-16 | 1.798e-17 |
| two layers gamma/epsilon | 1.421e-14 | 2.572e-16 |
| two layers epsilon/gamma | 7.105e-15 | 1.200e-16 |
| two layers alpha1_beta0/gamma | 0 | 0 |
| three layers mixed | 8.882e-15 | 4.202e-16 |

**Maximum relative error 4.2e-16**, entirely the official float64's own tail difference.

---

## 6. Reusable lessons (10 of them)

1. **The official package can be installed, and it is worth installing**.
   `uv pip install --python .venv-eval/bin/python --target /tmp/captum_pkgs --no-deps captum`
   → captum 0.9.0; then add `matplotlib` → `pillow` → `pyparsing/cycler/fonttools/kiwisolver/python-dateutil/contourpy`
   one by one. Installing into a `/tmp` target directory **does not pollute the judging
   environment**; the cross-check script runs with `PYTHONPATH=/tmp/captum_pkgs`.
   This is the key that gave this problem "official ground truth" instead of "guessing from code".

2. **`diff` the installed version against the main branch**: the differences in `lrp_rules.py` are
   only type annotations; the differences in `lrp.py` are only typing/pyre comments plus
   `log_usage(part_of_slo=True)`. The **semantics are identical**, so the installed version is
   safe to use for cross-checking.

3. **You must wrap the model in `nn.Sequential`**. When `nn.Linear` is the model directly,
   `model.children()` is empty → `self.layers` is empty → not a single rule is attached → all three
   rules give identical attributions. This is the earliest pitfall of this problem (in probe
   `probe843.py` the three rules produced identical output).

4. **`rule` is `del`-ed inside `_restore_model`** (`lrp.py::_remove_rules`), so if you want to
   inspect `layer.rule.relevance_output` after the run you must **instrument**
   (patch `_restore_model` or patch the hook factory functions).

5. **`GammaRule`'s gamma must be a float** (`self.gamma * module.weight.data.clamp(min=0)` is a
   tensor times a scalar). Passing a string raises
   `TypeError: only integer tensors of a single element can be converted to an index`.
   The problem side uses the exact rational string `"1/4"`; the cross-check script must convert
   it with `float(Fraction(...))`.

6. **The `outputs` captured by the hook is the value of the "second forward", while `inputs` is
   the activation of the "first forward"**. The two come from different places — this is the
   easiest thing to mix up.

7. **The backward chain propagates the previous layer's `rel_in`** (not `rel_out`). Measured hook
   trajectory: the `grad` seen by the `in` hook equals the previous layer's `rel_out @ W'`, and the
   `relevance` it returns equals that value × the original activation; this return value is
   precisely the `grad` that the next layer's `out` hook receives.

8. **`sign(0) = +1` needs a purpose-built example**. The anchor `W=[[1,1]], b=[-2], inputs=[1,1]`
   yields `outputs=[0]`, the denominator is `+eps`, `relevance_out = [1/eps] = [1e9]`, and
   `attribution = [0, 0]`. Dropping this rule raises `ZeroDivisionError`, and the mutation
   catches it every single time.

9. **The official delta and the IG delta have opposite signs**. LRP: `output - attr_sum`
   (`lrp.py`); IG: `attr_sum - (end_out - start_out)` (`attr/_utils/attribution.py`).
   Both problems live in the same collection, so the problem statement must state this
   explicitly — otherwise solvers will mix them up.

10. **Passing `upstream` explicitly as a parameter is safer than resetting it inside the
    function**. The first version wrote `one_hot(target)` inside `_layer_ledger`, so every layer
    reset it → correct for one layer, wrong for many layers (the official cross-check blew up on
    the two-layer case with `rel=0.219897`). The fix is to turn `upstream` into a parameter:
    the top layer gets the one-hot vector, the other layers get the previous layer's `rel_in`.
    **Cross-layer semantics must be validated by "an independent oracle + the official
    cross-check" double verification; printing the hook trajectory layer by layer is the fastest
    way to locate the bug.**

---

## 7. The three-layer verification structure of this problem

1. **Official layer**: run float64 directly with the installed captum `LRP`, cross-check the
   `attribution` (relative error ≤ 4.2e-16).
2. **Independent oracle layer**: write a separate `Fraction` implementation inside
   `test_cases.py` (a **second, independent code path** from the solution) and compare
   `activations` / `outputs` / `relevance_out` / `relevance_in` / `attribution` element by element.
3. **Mutation layer**: all 39 single-point mutations are CAUGHT (covering rule rewriting, sign/eps,
   the z-rule decomposition, the two-forward semantics, layer ordering, the delta sign, and
   illegal inputs).

**Real bugs caught by mutation (all of them are mistakes you would genuinely make while
writing this)**:

| Mutation | The error it caught |
|---|---|
| `stack-no-upstream-propagation` | resetting upstream per layer → broken cross-layer propagation |
| `stack-second-forward-chained` | writing the second forward as a chained forward with rewritten weights |
| `stack-attr-uses-rewritten-output` | using the rewritten output as the normalization baseline |
| `stack-relevance-order-reversed` | reversed layer order (rel_in[0] landing on the output layer) |
| `rewrite-gamma-scales-all` | implementing gamma as scaling all weights |
| `sign-zero-not-plus-one` | dropping sign(0) = +1 |

**Judging gaps (exposed by surviving mutations)**: `stack-empty-layers-allowed` and
`stack-out-of-range-target-allowed` survived at first — the tests did not cover "an empty layer
stack" nor "an out-of-range target". After adding the corresponding assertions everything is green.
**A surviving mutation is a judging gap** — this lesson is worth keeping.
