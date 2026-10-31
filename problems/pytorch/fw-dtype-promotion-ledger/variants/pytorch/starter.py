def solve(lhs, rhs, out=None, default_float="float32"):
    """按 PyTorch 类型提升规则算出二元算术的结果 dtype，并判断能否写回 out。

    参数
    ----
    lhs, rhs : 长度 2 的 (kind, dtype) 序列，代表 a ⊗ b（如 a * b）的两个操作数。
        kind ∈ {"tensor", "wrapped", "scalar"}：
            "tensor"   有维张量
            "wrapped"  0 维张量（由 Python 数包装而来，与有维张量语义不同）
            "scalar"   裸 Python 数
        dtype 的定义域随 kind 变化：
            "tensor"   12 个之一：bool uint8 int8 int16 int32 int64
                       float16 bfloat16 float32 float64 complex64 complex128
            "wrapped"  8 个之一：bool uint8 int8 int16 int32 int64 float32 float64
            "scalar"   3 个之一："bool" "int" "float"
    out : None 或 12 个 dtype 之一。None 表示只要结果 dtype；非 None 表示结果还要写回
        该 dtype 的输出张量，等价于 torch.add(a, b, out=torch.empty(..., dtype=out))。
    default_float : "float32" | "float64"，Python float 数被视作哪个 dtype
        （对应 torch.get_default_dtype()）。

    返回
    ----
    (promoted_dtype, status) 两个字符串：
        promoted_dtype : 12 个 dtype 之一，即 a ⊗ b 的结果类型。
        status         : out 为 None 时恒为 "ok"；否则写回被接受是 "ok"、
                         被拒绝是 "cast_error"。

    规则
    ----
    1. category 全序：bool < integral < floating < complex；uint8 属 integral。
    2. 有维张量无条件参与；"wrapped"/"scalar" 这一类只在它的 category **严格高于**
       所有有维张量的 category 时才参与，且此时结果只需装下该最高 category 的那些标量
       —— 低 category 的标量与低 category 的有维张量都不再约束结果。
       特例：两个操作数都不是有维张量时，0 维张量改按普通张量全量参与格内 join
       （w(uint8) ⊗ w(int8) -> int16），此时只有裸 Python 数继续走上面的严格更高规则
       （w(uint8) ⊗ py-int -> uint8，但 w(bool) ⊗ py-int -> int64）。
    3. "scalar" 先映射成 dtype 再参与规则 2：bool -> "bool"、int -> "int64"、
       float -> default_float。操作数的**值**永远不参与判定。
    4. 结果在最高 category 的格内取最小的「装得下所有参与该格的操作数」的类型：
       - integral：按 bool ⊂ uint8 ⊂ int8 ⊂ int16 ⊂ int32 ⊂ int64 的值域包含关系取，
         例如 uint8 ⊗ int8 -> int16、int8 ⊗ 0 维 int64 -> int8（后者是标量，不参与）。
       - floating：float16 与 bfloat16 **同层**（互不包含），单者保持自身，两者相遇 ->
         float32；float32 ⊗ float16 -> float32；float64 ⊗ 任何 -> float64。
       - complex：complex64 ⊗ complex128 -> complex128；参与集合里出现 float64（含
         default_float="float64" 的 Python float）也抬到 complex128，float32/整数不抬。
       - 整数从不放大浮点：int64 ⊗ float16 -> float16。

    写回门槛（out= 与 in-place 算术共用，三条禁令，其余一律放行哪怕截断）：
        - integral out 拒绝 floating / complex 结果；
        - bool out 只接受 bool 结果；
        - 非 complex out 拒绝 complex 结果。
       于是 int8 ← float32 是 "cast_error"，但 int8 ← int64、float16 ← float32、
       uint8 ← int16 都是 "ok"（会静默截断）。

    非法输入抛 ValueError：default_float 不在 {"float32", "float64"}；out 非 None 且
    不在 12 dtype 内；操作数不是长度 2 的 (str, str) 序列；kind 不属于三种；dtype 越出
    该 kind 的定义域。

    不得直接查表求解：禁止调用 torch.result_type / torch.promote_types / torch.can_cast
    / torch._promote_types（也不得用 torch.get_default_dtype() 之类去读取全局状态）。
    """
    raise NotImplementedError
