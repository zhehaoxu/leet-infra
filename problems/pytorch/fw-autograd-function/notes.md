# 详解：自定义 autograd.Function 的前反向账

## 出处

PyTorch 文档 *Extending PyTorch → Defining a custom Function*：自 1.8/1.9 起官方推荐**新版 API**——`forward` 不带 `ctx`，新增 `setup_context(ctx, inputs, output)` 负责 `save_for_backward`。文档明确指出旧式 `forward(ctx, ...)` 与 `setup_context` 混用会导致签名绑定错误（`signature.bind` 直接 `TypeError: missing a required argument`）。非 Tensor 输出（本例的 Python `int`）由引擎原样透传、不参与建图，其位置的反向梯度参数为 `None`；`ctx.needs_input_grad` 告知每个输入是否需要梯度，对不需要者返回 `None` 是文档规定的写法。`save_for_backward` 的张量默认在 backward 结束后释放（`retain_graph=False`），再次反传触发 RuntimeError。

## 核心与易错点

1. **两种 API 只能二选一**：定义了 `setup_context` 的类，`forward` 必须是 `forward(x, c)`；写成 `forward(ctx, x, c)` 会在 `.apply` 时抛 `TypeError`——这本身就是题面第 4 条约束的出处。
2. **不可微第二输出的正确姿势**：返回 Python `int` 时无需 `mark_non_differentiable`（它根本不是 Tensor，不进图）；若返回 Tensor 又忘标记，autograd 会把它当可微输出、backward 里被迫处理它的梯度。`type(neg_count).__name__ == "int"` 就是这条行为差的探针。
3. **`bwd_calls == 3` 的账**：apply 两次、第一次图反传 1 次；对第二个图连续 grad 两次，第二次**仍然进入了 backward**（入口计数器已加），是在取 `ctx.saved_tensors` 时才抛出「图已释放」的 RuntimeError。这解释了 `fwd_calls=2, bwd_calls=3` 的反直觉差值。
4. **手写 backward vs 对照 autograd**：解析式 `c*(2x*tanh + x²*(1-tanh²))*g` 与链式法则图的 ULP 级差异（末项 60.59123229980469 vs 60.591224670410156）说明为什么对拍要用 `allclose` 而不是 `torch.equal`。
5. **`ctx.c = c` 与 `save_for_backward` 的分工**：非输入派生的普通张量可以直接挂 ctx，但对**输入或输入 view** 必须走 `save_for_backward`，否则 autograd 无法检测内存复用/版本冲突。

## 对拍口径

判题 `_oracle` 是纯 Python `math` 层：用 `math.tanh` 逐元素重算 \(y\) 与解析梯度（float64），与 solution 的 float32 账对拍 `atol=rtol=1e-5`；`neg_count`、flags、计数器为硬编码精确比对。另设「int 输出不建图」「needs_input_grad 账」「重传报错」三条不变量断言。

## 复杂度

5 元素张量、两次 apply，微秒级；成本全在理解引擎行为。
