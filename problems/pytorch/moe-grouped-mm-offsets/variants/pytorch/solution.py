"""参考答案：一次同步取 `offs`，循环 E 组，每组三段切片矩阵乘。

这道题的分量全在边界口径上，四个点各对应一类典型错法：

1. **`offs` 是结束下标，不是组大小**。`count_e = offs[e] - offs[e-1]`，`offs` 本身是前缀和。
   把 `offs[e]` 直接当组大小用，第一组恰好相等看不出来，往后每组都错位；更隐蔽的是
   用 `offs[e]` 当区间起点（即当成 exclusive-prefix 的左端），那等于把每组整体右移一格。
2. **`out` / `grad_input` 必须先写 0 再逐段覆盖**，因为从 `offs[E-1]` 到 `S` 的这些行
   没有任何组会覆盖到。用 `torch.empty` 起手的实现，尾部行是未初始化内存：在 CPU 上常常
   碰巧是 0 而本地全绿，换一台机器或换个分配器就翻车。上游 fallback 写的正是
   `output = torch.zeros(...)`，这不是保守，是这条口径的实现。
3. **两段梯度的方向不一样，不能套同一个模板**。`grad_input` 是「rows×N 乘 N×K」，
   所以右乘 `weight[e].T`；`grad_weight` 要和 `weight` 同形（K×N），所以是
   `x.T @ grad_output`（「K×rows 乘 rows×N」）。写成 `grad_output.T @ x`
   会得到 N×K：K 与 N 不等时被形状检查拦下，相等时则要靠数值对拍；判题两种都放了。
   而“顺手”写成 `x @ grad_output.T`，形状照样是 K×N，
   数值却完全错位——这才是真正的陷阱。
4. **空组跳过，但要照常推进 `start`**。跳过只意味着不读 `weight[e]`、不写 `grad_weight[e]`，
   不意味着区间划分停摆。判题里有一条专门用例会验证这一点：把一个空专家的权重换成任意
   随机值，四个返回项必须逐位不变。

`counts` 由 `offs` 差分直接给出，不需要数真实行数。`offs.tolist()` 在循环外只做一次——
上游注释写得很直白，这是 single cpu<->gpu sync point，避免循环里反复同步。本参考答案
不自建计算图、不接 autograd：梯度只是被当作另一次矩阵乘来算。
"""

import torch


def _check_matrix(value, name, dims):
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor")
    if value.device.type != "cpu":
        raise ValueError(f"{name} must be on CPU")
    if value.dtype != torch.float64:
        raise ValueError(f"{name} must be float64")
    if value.dim() != dims or min(value.shape) < 1:
        raise ValueError(f"{name} must be a non-empty {dims}-D tensor")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")


def _check(x, weight, offs, grad_output):
    _check_matrix(x, "x", 2)
    _check_matrix(weight, "weight", 3)
    _check_matrix(grad_output, "grad_output", 2)
    if not isinstance(offs, torch.Tensor):
        raise ValueError("offs must be a torch.Tensor")
    if offs.device.type != "cpu":
        raise ValueError("offs must be on CPU")
    if offs.dtype not in (torch.int32, torch.int64):
        raise ValueError("offs must be an integer tensor (int32 or int64)")
    if offs.dim() != 1 or offs.numel() < 1:
        raise ValueError("offs must be a non-empty 1-D [E] tensor")
    if offs.size(0) != weight.size(0):
        raise ValueError("offs length must match the number of experts")
    if x.size(1) != weight.size(1):
        raise ValueError("x.size(1) must match weight.size(1)")
    if grad_output.size(1) != weight.size(2):
        raise ValueError("grad_output.size(1) must match weight.size(2)")
    if grad_output.size(0) != x.size(0):
        raise ValueError("grad_output.size(0) must match x.size(0)")
    ends = offs.tolist()
    previous = 0
    for end in ends:
        if end < 0:
            raise ValueError("offs entries must be non-negative")
        if end < previous:
            raise ValueError("offs must be non-decreasing")
        previous = end
    if previous > x.size(0):
        raise ValueError("the last offs entry must not exceed the number of rows")


def solve(x, weight, offs, grad_output):
    _check(x, weight, offs, grad_output)
    rows, k_dim = x.shape
    experts = weight.size(0)
    n_dim = weight.size(2)

    # 尾部 [offs[-1], rows) 不属于任何专家，靠这里的 0 初始化交出全零行
    out = torch.zeros(rows, n_dim, dtype=torch.float64)
    grad_input = torch.zeros(rows, k_dim, dtype=torch.float64)
    grad_weight = torch.zeros(experts, k_dim, n_dim, dtype=torch.float64)
    counts = torch.zeros(experts, dtype=torch.int64)

    ends = offs.tolist()  # 循环外一次性取到 host，避免逐组同步
    start = 0
    for expert, end in enumerate(ends):
        counts[expert] = end - start
        if start != end:
            x_slice = x[start:end]
            g_slice = grad_output[start:end]
            torch.mm(x_slice, weight[expert], out=out[start:end])
            torch.mm(g_slice, weight[expert].t(), out=grad_input[start:end])
            torch.mm(x_slice.t(), g_slice, out=grad_weight[expert])
        start = end

    return out, grad_input, grad_weight, counts
