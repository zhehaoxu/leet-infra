import torch


def solve(spec, *operands):
    """einsum-reduce：按标注串求值，语义与 torch.einsum 一致（受限子集）。

    入参：
    - spec: str，形如 "bij,bjk->bik"，逗号分隔各输入标注，"->" 右侧为输出标注。
      标注只允许单个字母（a–z / A–Z），空格忽略。
    - operands: 与 spec 左侧数量相同的张量，float32（内部可用 float64 累计，返回值转回 float32）。

    支持范围（超出即抛 ValueError）：
    - 必须显式写出 "->"（省略输出标注的形式不支持）。
    - 不支持省略号 "..."。
    - 同一个操作数的标注里同一字母不得重复（即不支持 trace / diagonal）。
    - 输出标注不得重复，且每个输出字母必须在某个输入中出现。
    - 输入字母若不出现在输出中，就是被求和归约掉的轴（contracted）。
    - 操作数个数 1–3 个，每个操作数的维数必须与其标注字母数相等。

    语义：先把所有操作数按统一字母顺序对齐（缺失的轴补成长度 1），逐元素相乘，
    再对被归约的轴求和，最后把剩余轴按输出标注的顺序排布。返回形状为各输出轴长度依次的张量，
    dtype 与输入相同（float32）。纯归约（如 "ij->"）返回 0 维张量。
    """
    # TODO
    raise NotImplementedError
