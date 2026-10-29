import torch


def solve(x, p=2.0, dim=None, keepdim=False):
    """沿 dim 算 p 阶范数，但必须「先按段内最大幅值缩放、再归约」，避免平方时的上溢与下溢。

    不许调用 torch.norm / torch.linalg.norm / x.norm() / torch.linalg.vector_norm，
    也不许先算 |x|**p 再求和再开 p 次方——那是「错但大多数用例能过」的写法：
    torch.linalg.norm([3e154, 4e154]) 给 inf，torch.linalg.norm([3e-200, 4e-200]) 给 0.0。

    入参：
    - x: float32 或 float64 张量，输出 dtype 跟输入一致。非 Tensor 或非浮点抛 ValueError。
    - p: int 或 float（bool 不算数）。必须 > 0，允许 math.inf（切比雪夫范数）。
      NaN、<= 0、-inf、非数值类型都抛 ValueError。
    - dim: None 表示整体归约，返回 0-D 张量（此时 keepdim 不起作用）；
      否则为 int（bool 不算），允许负轴，归一化后不在 [0, ndim) 内抛 ValueError。
    - keepdim: 只接受 bool，True 时保留被归约的轴（长度记为 1）。

    核心恒等式（seg 是一个待归约段，m = max_i |x_i|）：
        ||x||_p = m * ( Σ_i (|x_i| / m)^p )^(1/p)
    缩放后每一项都落在 [0, 1]，求幂既不会上溢也不会下溢，最后再把 m 乘回去。
    p == inf 时直接返回 m——幂次对 max 没有意义，而这个 m 正是缩放因子本身。

    特殊情形按优先级从上往下判（先命中先定值，和 torch 归约的 NaN 传播一致）：
    1. 段内出现 NaN        -> NaN（NaN 优先于 inf）
    2. 段内出现 ±inf       -> +inf
    3. m == 0（整段是 0）  -> 0.0（否则 |x|/m 变成 0/0）
    4. 段长度为 0（空张量或被归约轴长度为 0）-> 0.0
    5. 其余走上面的缩放公式

    中间量一律用 float64 累加，最后再转回输入的 dtype（float32 输入 -> float32 输出）。
    返回值是一个张量：float32 输入时 5e154 这类超出 float32 范围的结果自然转成 inf，不算错。
    """
    # TODO
    raise NotImplementedError
