import torch


def solve(logits, bias, k, tau):
    """中文契约 / English contract
    logits [T,E]：CPU float64 路由 logits，未归一化 / CPU float64 router logits, not normalized.
    bias：None，或 CPU float64 [E]；**只参与选择，不参与加权** / None or CPU float64 [E]; selection-only, never in weights.
    k：真正的 Python int 且 1<=k<=E；bool、浮点、越界均 ValueError / a true int with 1<=k<=E, else ValueError.
    tau：非负有限的 int/float；tau=0 表示关闭软限幅；bool、NaN、±inf、负数 ValueError /
        non-negative finite int/float; 0 disables the clamp; bool/NaN/inf/negative raise ValueError.
    规则 / rules：c = tau*tanh(logits/tau) 若 tau>0，否则 c = logits；s = sigmoid(c)；
        每行按 s+bias 降序稳定取 top-k（并列取专家编号小者）/ clamped sigmoid scores,
        stable descending top-k over s+bias with ties broken by the smaller expert index;
        w = s[选中]/Σ s[选中]（用不带 bias 的 s 归一化）；若选中项之和下溢成 0，则 w = 1/k /
        renormalize the **unbiased** s over the picked slots; fall back to uniform 1/k when that sum is 0.
    返回 / Returns (weights [T,k] float64, indices [T,k] int64, scores [T,E] float64)。
    形状非法、dtype 非 float64、非 CPU、bias 长度不等于 E、非张量输入都抛 ValueError；不修改输入，接受非连续张量 /
        bad shapes, non-float64, non-CPU, bias length mismatch and non-tensors raise ValueError; mutate nothing.
    """
    raise NotImplementedError
