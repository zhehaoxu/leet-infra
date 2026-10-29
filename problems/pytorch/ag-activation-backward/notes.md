# 详解：折点约定与「前向用什么近似」的连带后果

## 出处

「说一下 GELU 的导数」是大模型框架面试的常规快问，因为 Transformer 的 FFN 几乎都用它；`relu` 在 0 处取什么则是考察是否理解**次梯度**而非「导数」。把八个激活放在一题里，正是面试现场最常见的追问节奏：先 relu，再 leaky，再 elu，最后 gelu/silu。

## 折点为什么可以「随便取」而又不能乱取

\(x=0\) 处 relu 不可微，autograd 用的是右导数约定（`torch.relu` 的 backward 是 `where(x > 0, 1, 0)`），所以 0 处给 0。这不影响收敛性（单点概率为 0），但**影响对拍**：写 `x >= 0` 的用例在包含 0 的输入上会逐位不同。elu 反而在 0 处连续可导（左极限 \(e^0=1\)、右极限 1），silu/softplus 都是解析函数，取值 0.5 是真实导数而非约定。

## gelu 的两种形式不可混用

精确式 \(x\,\Phi(x)\)，\(\Phi(x)=\tfrac12(1+\operatorname{erf}(x/\sqrt2))\)，导数 \(\Phi(x)+x\phi(x)\)，\(\phi\) 是标准正态密度。tanh 近似 \(\tfrac12x(1+\tanh(0.7978( x+0.044715x^3)))\) 的导数形式完全不同。PyTorch 的 `F.gelu` 默认用精确式（`approximate='none'`），所以判题按 erf 对拍；若实现里手写了 tanh 近似，会在 \(|x|\approx2\) 处差出 \(10^{-3}\) 量级而挂掉。

## 核心与易错点

- **不要用 `torch.sigmoid(x)*(1-torch.sigmoid(x))` 去算 softplus 的导数**：softplus 的导数就是 \(\sigma(x)\)，多乘一次立即错。
- **`leaky_relu` 的折点**：`F.leaky_relu` 的 backward 实测是 `x > 0 ? 1 : slope`，所以 \(x=0\) 返回 `negative_slope`（默认 0.01），与 relu 在 0 处给 0 的处理方式一致。
- **形状与 dtype**：`x` 是标量张量也要返回同 dtype；bool 掩码类比较运算在 float64 下转 float 再乘 `dY`。
- **`kind` 校验先于计算**：非法 kind 必须抛 ValueError，而不是静默走 relu 分支。

## 与邻近题的区别

`ag-layernorm-backward`/`ag-rmsnorm-backward` 是归约耦合的反向，本题严格逐元素、无归约；`ag-lstm-bptt-backward` 里同样要用到 sigmoid/tanh 的导数，那里错一个因子会让整条时间线崩掉。

## 对拍口径与复杂度

oracle 逐个 `F.{relu,leaky_relu,elu,sigmoid,tanh,gelu,silu,softplus}` 建图取梯度，float64 atol=1e-9。示例只用 relu 与五个特殊点（含 0）。复杂度 \(O(\text{numel})\)，八个分支互不共享中间量。
