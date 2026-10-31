# 详解：类型提升账——算术结果 dtype 与写回门槛

## 出处

pytorch/pytorch `docs/source/tensor_attributes.md`（锚点 `(type-promotion-doc)=`）：`:106-118` 是三条提升规则（scalar/0 维张量只在 category 严格更高时才参与、否则只由有维操作数决定），`:120-123` 说明「Python float 视作 `torch.get_default_dtype()`、非布尔整数视作 `torch.int64`、且不检查值」，`:165-168` 是三条写回禁令，`:170-188` 给出允许/禁止的 in-place 清单。教程侧对应 *Learn the Basics* 的 tensor/dtype 段落。

两处文档已经落后于实现，本题按 torch 2.14.1 实测语义出题：`:122` 的「Complex types are not yet supported」不成立（复数有自己的两档格），且 `:114-116` 只写了「0 维张量 category 更高时」，没写「所有操作数都是 0 维/裸数时」这一分支。

## 核心与易错点

1. **wrapped number 的「严格更高」是最容易写反的一条**：`float16 张量 ⊗ 2.0 → float16`，`default_float` 完全不介入——因为 Python float 与 float16 同属 floating，不严格更高。这正是「为什么我的 half 乘法没升到 float」这类问题的标准答案。
2. **uint8 属于 integral，不因为无符号就更宽**：`int8 ⊗ uint8 → int16`——要在值域上同时装下 [-128,127] 与 [0,255]，最小的整数类型是 int16，而不是 int8 或 uint8。
3. **整数从不放大浮点**：`int64 ⊗ float16 → float16`（文档 `:159-162` 的 `torch.add(long_tensor, float_tensor).dtype == float32` 同理）。跨 category 时低 category 的操作数被完全忽略。
4. **float16 与 bfloat16 同层**：不存在「bfloat16 ⊃ float16」的包含关系，单者保持自身，两者同时出现才升 float32——把浮点当成一条线性链是最常见的错法。
5. **全 0 维的隐藏分支**（文档未写、实测得来）：没有有维张量时 0 维张量不再降级为 wrapped，而是像普通张量一样全量参与 join，所以 `0 维 uint8 ⊗ 0 维 int8 → int16`；裸 Python 数继续走严格更高，所以 `0 维 uint8 ⊗ 3 → uint8`、`0 维 bool ⊗ 3 → int64`。
6. **提升结果与写回门槛是两件事**：`uint8 ← int16` 依然放行（文档 `:181` 的 `uint_tensor *= int_tensor` 是官方举例），静默截断自负；三条禁令只挡跨 category 的方向（整出、布尔出、非复数出）。
7. **`copy_` 不走这个门槛**：`torch.zeros(3, torch.uint8).copy_(float32)` 成功，`torch.add(..., out=uint8)` 抛 `RuntimeError: result type Float can't be cast to the desired output type Byte`——「赋值能过、算术不能过」是常态。
8. **复数格的抬升只由参与集合里的 float64 触发**：`complex64 ⊗ float64 张量 → complex128`，但 `complex64 ⊗ 0 维 float64 → complex64`（后者不严格高于 complex），`complex64 ⊗ 2.0` 无论 `default_float` 是什么都保持 complex64。

## 定义域外的真实行为（刻意排除）

- 0 维张量若允许 float16/bfloat16/complex，会得到文档里没有的 dtype：实测 `torch.ones(3, float16) + torch.ones((), complex64)` → `torch.complex32`，bfloat16 同理 → `torch.bcomplex32`。这两个类型不在 12 个公开 dtype 里，所以题面把 wrapped 的定义域限制为 8 个（bool/5 个整数/float32/float64）。
- Python complex 只在 `result_type(number, number)` 与 `result_type(number, tensor)` 两个重载里可用，`torch.result_type(1j)` 单参数直接 TypeError，所以标量域只有 `bool/int/float`。
- `torch.result_type` 只接受**恰好两个**位置参数（无 varargs、不接受列表，实测 TypeError），所以本题刻意设计成二元操作数契约：既没有折叠次序歧义，也让每一格都能一对一回源。

## 对拍口径

solution 是纯 Python 的规则推演，不调用任何 torch 提升 API。`test_cases.py` 直接拿 torch 当 oracle：

- 提升网格 23×23（12 个有维 dtype + 8 个 0 维 dtype + 3 个标量，两侧同域）× {float32, float64} 两种 `default_float` = 1058 组，逐组对 `torch.result_type`（用 `torch.set_default_dtype` 临时切换并在 `__exit__` 还原）。
- 写回网格 276 组操作数（有维/0 维/标量 × 12 个有维）× 12 个 out dtype = 3312 组，逐组对 `torch.add(a, b, out=)` 是否抛 `RuntimeError`。
- 操作数的值一律取 1，避免触发与 dtype 无关的值溢出检查。

题面 24 行示例由参考实现与 torch 双向实测（脚本 `ex.py`），逐行反解回 `solve` 校验通过。

## 复杂度

每次调用 O(1)；全量对拍 4370 次 torch 调用，本机 < 1 秒。
