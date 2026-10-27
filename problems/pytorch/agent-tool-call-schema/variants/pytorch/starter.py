def solve(schema, arguments):
    """把模型给出的工具调用参数按 JSON Schema 校验并归一化。

    schema:{"type": "object", "properties": {...}, "required": [...],
            "additionalProperties": bool}，properties 的每一项是子 schema；
    arguments:模型产出的参数字典（可能缺字段、类型不对、多字段）。

    规则：
    1) required 中的字段缺失 → "$.<name>: missing required property"
       （若该字段带 default，则按默认值补齐、不算缺失）；
    2) 类型：string/boolean 严格；integer 接受 int 与「值为整数的 float」并归一成 int，
       拒绝 bool；number 接受 int/float 并归一成 float，拒绝 bool；
       array 走 items 递归（错误路径 $.tags[i]）；object 走 properties/required 递归
       （错误路径 $.meta.days）；
    3) enum：按「类型 + 值」双重要求匹配（所以 enum=[0, 1] 时 True 不算命中），
       不命中 → "$.<name>: value <repr> is not one of <list>";
    4) 校验失败的字段不写入归一结果；缺失且非 required 的字段若有 default 则填入；
    5) additionalProperties=False 时，properties 之外的字段 →
       "$.<name>: unexpected property"（按 arguments 的插入顺序），否则原样保留。

    错误顺序：先 required 缺失，再按 properties 插入顺序，最后是多余字段。
    返回 (normalized, errors)。
    """
    # TODO
    raise NotImplementedError
