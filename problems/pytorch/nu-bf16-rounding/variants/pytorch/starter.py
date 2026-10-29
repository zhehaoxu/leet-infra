import torch


def solve(x, rounding='rne'):
    """按位把 float32 模拟成 bfloat16 能表示的数，结果仍以 float32 返回（fake cast）。

    不许调用 x.to(torch.bfloat16) / x.bfloat16()，也不许借 numpy 或 struct 做类型转换——
    考点就是自己把舍入写成位运算。

    入参：
    - x: float32 张量，任意形状（非连续可以先 contiguous，不算错误）。其它 dtype 抛 ValueError。
    - rounding: 'rne' 就近取偶 | 'rz' 向零截断 | 'rna' 就近、ties 远离零。其它值抛 ValueError。

    位级定义：bfloat16 就是 float32 的高 16 位，丢掉低 16 位。设 u 为 32 位模式按无符号整数读出，
    L = u & 0xFFFF 是被丢掉的部分，第 16 位（u >> 16 的最低位）是保留下来的尾数末位：
    - 'rz':  直接 u & 0xFFFF0000
    - 'rne': L > 0x8000 时进一位；L == 0x8000（正好是 ties）时看第 16 位，为 1 才进位（取偶）；
             其余不进位。等价于 u + 0x7FFF + ((u >> 16) & 1) 后取高 16 位。
    - 'rna': L >= 0x8000 就进位（等价于 u + 0x8000 后取高 16 位），ties 一律远离零。
    进位会顺着尾数溢出到指数位，自然把过大的值推到 ±inf，不需要额外判断溢出。

    特殊值：
    - ±0 保留符号位；±inf 原样返回。
    - NaN：保留符号位，并把保留下来的尾数最高位（float32 的第 22 位）强制置 1，
      保证结果仍是 NaN——裸加进位会把某些 NaN 舍成 ±inf，这是本题最容易踩的坑。
    - 已经在 bf16 网格上的值，三种模式都原样返回（幂等）。
    """
    # TODO
    raise NotImplementedError
