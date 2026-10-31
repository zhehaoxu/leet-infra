# 类型提升账：算术结果 dtype 与写回门槛

## 题目描述

`int8 张量 * uint8 张量` 出来的不是 int8；`float16 张量 * 2.0` 并不会升到 float32；而 `torch.add(a, b, out=c)` 里 c 的 dtype「装不下」结果时直接抛 `result type Float can't be cast to the desired output type Char`。这三种行为都写在官方 *Tensor attributes* 页的 Type Promotion Mechanics 一节，属于 PyTorch 入门教程里 dtype / Tensor 属性章节的必读内容，也是调试混合精度与量化推理时最容易踩的静默坑。

本题把这套规则抽象成一张纯 dtype 的账：给定二元算术两个操作数的种类与 dtype，再给一个可选的输出 dtype，要求算出结果 dtype 以及「能否写回」，全程不涉及任何张量数值。

## 规则

实现 `solve`（参数与非法输入的契约见 starter docstring）。要点：

1. category 全序：`bool < integral < floating < complex`。uint8 属于 integral，不因为「无符号」就更宽。
2. 只有**裸 Python 数与 0 维张量**（文档称 wrapped number）享受「严格更高才参与」：它们的 category 不严格高于所有有维张量时对结果毫无影响——于是 `int8 ⊗ float16 张量` 之外还有 `float16 张量 ⊗ 2.0 → float16`、`uint8 张量 ⊗ 3 → uint8`、`int8 张量 ⊗ 0 维 int64 → int8`。一旦严格更高，结果只需装下这些数（`bool 张量 ⊗ 2.0 → float32`），此时连低 category 的有维张量也不再约束结果。
3. 两个操作数都不是有维张量时，0 维张量改按普通张量全量参与（`0 维 uint8 ⊗ 0 维 int8 → int16`），继续走「严格更高」规则的只剩裸 Python 数（`0 维 uint8 ⊗ 3 → uint8`、`0 维 bool ⊗ 3 → int64`）。
4. 值永远不参与判定：Python int 视作 int64，Python float 视作 `default_float`（对应 `torch.get_default_dtype()`）。
5. 结果在最高 category 的格内取「最小的装得下者」：整数按值域包含链（`int8 ⊗ uint8 → int16`）；浮点里 float16 与 bfloat16 **同层**，单者保持自身、相遇升 float32；整数从不放大浮点（`int64 ⊗ float16 → float16`）；复数格只被**参与集合里的** float64 抬高（`float64 ⊗ complex64 → complex128`，但 `complex64 ⊗ 0 维 float64 → complex64`）。
6. 写回门槛（`out=` 与 in-place 算术共用）三条禁令：整数 out 拒绝 floating/complex 结果；bool out 只接受 bool 结果；非复数 out 拒绝复数结果。其余一律放行，哪怕静默截断（`int8 ← int64`、`float16 ← float32` 都是 `ok`）。

## 示例

每行 `lhs rhs out default_float -> 结果`；`w`=wrapped（0 维张量）、`t`=tensor（有维张量）、`s`=scalar（裸 Python 数）。

```
('t','int8')   ('t','uint8')  None 'float32' -> ('int16',   'ok')
('t','int64')  ('t','float16') None 'float32' -> ('float16', 'ok')
('t','float16') ('t','bfloat16') None 'float32' -> ('float32', 'ok')
('t','bool')   ('s','int')    None 'float32' -> ('int64',   'ok')
('t','bool')   ('s','float')  None 'float32' -> ('float32', 'ok')
('t','uint8')  ('s','int')    None 'float32' -> ('uint8',   'ok')
('t','uint8')  ('s','float')  None 'float32' -> ('float32', 'ok')
('t','int8')   ('w','int64')  None 'float32' -> ('int8',    'ok')
('t','float16') ('s','float') None 'float32' -> ('float16', 'ok')
('t','float16') ('s','float') None 'float64' -> ('float16', 'ok')
('t','float64') ('s','float') None 'float64' -> ('float64', 'ok')
('w','uint8')  ('w','int8')   None 'float32' -> ('int16',   'ok')
('w','uint8')  ('s','int')    None 'float32' -> ('uint8',   'ok')
('w','bool')   ('s','int')    None 'float32' -> ('int64',   'ok')
('s','bool')   ('s','int')    None 'float32' -> ('int64',   'ok')
('t','float64') ('t','complex64') None 'float32' -> ('complex128', 'ok')
('t','complex64') ('w','float64') None 'float32' -> ('complex64',  'ok')
('t','int8')   ('t','float32') 'int8'  'float32' -> ('float32',  'cast_error')
('t','int8')   ('t','int64')   'int8'  'float32' -> ('int64',    'ok')
('t','uint8')  ('t','int8')    'uint8' 'float32' -> ('int16',    'ok')
('t','float32') ('t','float32') 'float16' 'float32' -> ('float32', 'ok')
('t','bool')   ('s','int')     'bool'  'float32' -> ('int64',    'cast_error')
('t','complex64') ('t','float64') 'float64' 'float32' -> ('complex128', 'cast_error')
('t','float64') ('t','float64') 'complex64' 'float32' -> ('float64', 'ok')
```

推演第 8 行：`('t','int8') ⊗ ('w','int64')` 里 0 维 int64 与有维 int8 同属 integral，category 不严格更高 → 不参与 → 结果只由 int8 决定。第 12 行反过来：没有有维张量，0 维 uint8 与 0 维 int8 全量参与，最小的装得下者是 int16。

## 约束

- dtype 一律用 12 个小写字符串之一：`bool uint8 int8 int16 int32 int64 float16 bfloat16 float32 float64 complex64 complex128`；`status` 只能是 `"ok"` 或 `"cast_error"`。
- 纯离散量精确对拍，无浮点比较、无随机数。
- 不得调用 `torch.result_type` / `torch.promote_types` / `torch.can_cast` 查表求解。
