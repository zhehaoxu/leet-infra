# VITS 归一化流前向/反向与 log-det 账

## 背景

VITS 是一种端到端语音合成模型。它的声学模型里有一大块**归一化流（normalizing flow）**：把一串高斯噪声通过若干可逆变换映射成声学隐变量，再用它的逆变换把隐变量映射回噪声。训练时靠「变换前后的联合概率密度之差」算似然，所以每个可逆变换除了输出，还必须报出它的 **log-det**（行列式对数）。

VITS 的这套 flow 全部写在一个仓库里的 `modules.py` 与 `models.py`：

- `Log`：逐元素取对数（带下界截断）。
- `Flip`：把 channel 维反转。
- `ElementwiseAffine`：逐 channel 的仿射变换。
- `ResidualCouplingLayer`：把 channel 一分为二，一半直通、另一半做仿射（coupling）。
- `ResidualCouplingBlock`：把若干 coupling 和 flip 串成一个 flow 序列。
- `StochasticDurationPredictor`：可微的时长预测器，它的 NLL 里同时含 `nll` 与 `logq` 两笔账。

本题要求你用纯 Python 重建这几笔账，并把官方实现里若干**不看源码就会写错**的细节钉死。

## 口径约定

- 所有能走精确有理数（内部按 `fractions.Fraction`）的部分一律走 `Fraction`；`log` / `exp` 本身是无理数，按 `float` 计算。**最终所有出参统一转成 `float`。**
- 三维张量的形状统一记作 `(B, C, T)`：`B` 是 batch，`C` 是 channel，`T` 是时间步。
- `x` 是嵌套列表，按 `(B, C, T)` 展平成一维后参与计算；展平后长度必须等于 `B*C*T`，否则报错。
- `mask` 官方叫 `x_mask`。官方在不同地方要求两种形状：
  - coupling / block 用的 `x_mask` 形状是 `(B, 1, T)`，对 channel 维**广播**；
  - `Log` / `ElementwiseAffine` 要求 mask 与 `x` **同形**（`(B, C, T)`）。

  本题统一接受这两种形状：给定的 mask 若展平后长度为 `B*T`，按 `(B, 1, T)` 广播到 `(B, C, T)`；若长度为 `B*C*T`，按 `(B, C, T)` 直接使用。两种都不匹配则报错。
- `m` / `logs` 是**逐 channel** 的参数：可以是标量（广播到所有 channel）、长度为 `C` 的一维列表，或 `(C, 1)` 的嵌套列表。
- 所有非法输入一律抛 `ValueError`（例如 `shape` 不是 3 个正整数、`x` 长度不匹配、coupling 的 channel 数为奇数、`m` 长度不匹配等）。
- `reverse=True` 时走官方的反向分支。

## mode 1：`flow_primitive` —— 单个 flow 基元

```python
solve("flow_primitive", kind="log", shape=[1, 1, 3],
      x=[[[0.5, 2.0, 4.0]]], mask=[[[1.0, 1.0, 1.0]]])
```

签名：

```python
solve("flow_primitive", kind, shape, x, mask, m=None, logs=None, reverse=False)
```

返回：

```python
{"y": [...], "logdet": [...]}   # y 展平成 B*C*T；logdet 每个 batch 一个
```

`kind` 有三种：

### `kind="log"`

官方 `modules.py::Log.forward`：

```python
def forward(self, x, x_mask, reverse=False, **kwargs):
    if not reverse:
      y = torch.log(torch.clamp_min(x, 1e-5)) * x_mask
      logdet = torch.sum(-y, [1, 2])
      return y, logdet
    else:
      x = torch.exp(x) * x_mask
      return x
```

要点：

- 前向是 `y = log(clamp_min(x, 1e-5)) * mask`：**先截断到 `1e-5` 再取对数**，然后整体乘 mask。
- `logdet = sum(-y)`，对 channel 与 time 两维求和、每个 batch 一个。**mask 已经乘进 `y`**，所以被 mask 掉的位置对 logdet 贡献 `0`。
- 反向是 `x = exp(x) * mask`：**反向也要乘 mask**。

### `kind="flip"`

官方 `modules.py::Flip.forward`：

```python
def forward(self, x, *args, reverse=False, **kwargs):
    x = torch.flip(x, [1])
    if not reverse:
      logdet = torch.zeros(x.size(0)).to(dtype=x.dtype, device=x.device)
      return x, logdet
    else:
      return x
```

要点：

- ★★ 翻的是 **channel 轴**（`torch.flip(x, [1])`），**不是**时间轴。`(1, 2, 3)` 形状的输入翻出来是 `(3, 2, 1)`。
- ★ `Flip.forward` **完全忽略 `x_mask`**（签名里根本没有 mask 参数），mask 为 0 的位置照样翻。
- `logdet` 恒为 `0`；反向也是 flip（自逆）。

### `kind="affine"`

官方 `modules.py::ElementwiseAffine.forward`：

```python
def forward(self, x, x_mask, reverse=False, **kwargs):
    if not reverse:
      y = self.m + torch.exp(self.logs) * x
      y = y * x_mask
      logdet = torch.sum(self.logs * x_mask, [1,2])
      return y, logdet
    else:
      x = (x - self.m) * torch.exp(-self.logs) * x_mask
      return x
```

要点：

- 前向 `y = (m + exp(logs) * x) * mask`：**mask 作用在整体上**，被 mask 掉的位置输出 `0`。
- `logdet = sum(logs * mask)`，对 channel 与 time 求和、每个 batch 一个。
- 反向 `x = (x - m) * exp(-logs) * mask`：**mask 同样作用在整体上**。
- 官方 `m` / `logs` 是 `nn.Parameter(torch.zeros(channels, 1))`，即形状 `(channels, 1)`。

## mode 2：`coupling_layer` —— 单层 coupling

```python
solve("coupling_layer", shape=[1, 4, 2],
      x=[[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]],
      mask=[[[1.0, 1.0]]], m=[[[0.1, 0.2], [0.3, 0.4]]])
```

签名：

```python
solve("coupling_layer", shape, x, mask, m, logs=None, mean_only=True, reverse=False)
```

官方 `modules.py::ResidualCouplingLayer.forward`（本题把 `pre`/`enc`/`post` 换成直接给定 `m` / `logs`）：

```python
def forward(self, x, x_mask, g=None, reverse=False):
    x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
    h = self.pre(x0) * x_mask
    h = self.enc(h, x_mask, g=g)
    stats = self.post(h) * x_mask
    if not self.mean_only:
      m, logs = torch.split(stats, [self.half_channels] * 2, 1)
    else:
      m = stats
      logs = torch.zeros_like(m)
    if not reverse:
      x1 = m + x1 * torch.exp(logs) * x_mask
      x = torch.cat([x0, x1], 1)
      logdet = torch.sum(logs, [1, 2])
      return x, logdet
    else:
      x1 = (x1 - m) * torch.exp(-logs) * x_mask
      x = torch.cat([x0, x1], 1)
      return x
```

要点：

- ★ channel 必须为偶数（官方 `assert channels % 2 == 0`），`half = C // 2`。
- `x0` 是前一半 channel，**原样透传**（官方不 mask `x0`）；`x1` 是后一半 channel。
- `m` 的长度必须是 `B * half * T`。
- `mean_only=True`（官方 block 里的用法）时 `logs` 恒为 `0`，此时 logdet 恒为 `0`。
- `mean_only=False` 时必须给 `logs`，长度与 `m` 相同。
- ★★ 前向是 `x1 = m + x1 * exp(logs) * x_mask`：**mask 只乘在 `x1` 项上，`m` 不被 mask** → mask 为 0 的位置，输出是 **`m`**（不是 `0`）。
- ★★ `logdet = sum(logs, [1, 2])`：**没有乘 mask**。即使某些位置的 mask 是 0，那些位置的 `logs` 照样计入 logdet。
- 反向 `x1 = (x1 - m) * exp(-logs) * x_mask`（mask 乘在整体上）。

## mode 3：`coupling_block` —— flow 编排账

```python
solve("coupling_block", shape=[1, 4, 2],
      x=[[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]],
      mask=[[[1.0, 1.0]]],
      m_list=[[[[0.1, 0.2], [0.3, 0.4]]], [[[0.5, -0.25], [0.125, 0.75]]]])
```

签名：

```python
solve("coupling_block", shape, x, mask, m_list, reverse=False)
```

官方 `models.py::ResidualCouplingBlock`：

```python
self.flows = nn.ModuleList()
for i in range(n_flows):
  self.flows.append(modules.ResidualCouplingLayer(..., mean_only=True))
  self.flows.append(modules.Flip())

def forward(self, x, x_mask, g=None, reverse=False):
    if not reverse:
      for flow in self.flows:
        x, _ = flow(x, x_mask, g=g, reverse=reverse)
    else:
      for flow in reversed(self.flows):
        x = flow(x, x_mask, g=g, reverse=reverse)
    return x
```

要点：

- flow 序列固定是 `[Coupling_0, Flip, Coupling_1, Flip, ...]`，长度是 `2 * len(m_list)`。
- ★ 前向按 `flows` 的**正序**执行，反向按**逆序**执行（`reversed(self.flows)`）。顺序不能换：coupling 与 flip 不满足交换律。
- 每个 coupling 都是 `mean_only=True` → `logs` 恒为 `0` → **整个 block 的 logdet 恒为 `0`**。
- ★ 官方 `forward` 里 logdet 被丢弃（`x, _ = flow(...)`），反向分支甚至只返回 `x`。本题统一返回累计 logdet（全为 `0`）。
- `m_list` 不能为空。

## mode 4：`duration_nll` —— 时长预测器的 NLL 与 logq 账

```python
solve("duration_nll", w=[[[2.0, 3.0, 4.0]], [[1.5, 2.5, 0.5]]],
      e_q=[[[0.1, -0.2, 0.3], [0.4, -0.5, 0.6]],
            [[0.7, 0.8, -0.9], [1.0, 0.2, -0.3]]],
      z_u=[[[0.25, -0.5, 0.75]], [[0.1, 0.2, 0.3]]],
      z1=[[[0.11, -0.22, 0.33]], [[0.44, -0.55, 0.66]]],
      mask=[[[1.0, 1.0, 1.0]], [[1.0, 0.0, 1.0]]],
      post_logdets=[0.5, -0.25], flow_logdets=[1.25, 0.75])
```

签名：

```python
solve("duration_nll", w, e_q, z_u, z1, mask, post_logdets, flow_logdets)
```

官方 `models.py::StochasticDurationPredictor.forward`（本题把随机量与 flow 内部结构换成给定值）：

```python
logdet_tot_q = 0
e_q = randn(B,2,T) * x_mask
z_q = e_q
for flow in self.post_flows:
  z_q, logdet_q = flow(z_q, x_mask, g=(x + h_w))
  logdet_tot_q += logdet_q
z_u, z1 = torch.split(z_q, [1, 1], 1)
u = torch.sigmoid(z_u) * x_mask
z0 = (w - u) * x_mask
logdet_tot_q += sum((logsigmoid(z_u) + logsigmoid(-z_u)) * x_mask, [1, 2])
logq = sum(-0.5 * (log(2*pi) + e_q**2) * x_mask, [1, 2]) - logdet_tot_q

logdet_tot = 0
z0, logdet = self.log_flow(z0, x_mask)
logdet_tot += logdet
z = torch.cat([z0, z1], 1)
for flow in flows:
  z, logdet = flow(z, x_mask, g=x, reverse=reverse)
  logdet_tot = logdet_tot + logdet
nll = sum(0.5 * (log(2*pi) + z**2) * x_mask, [1, 2]) - logdet_tot
return nll + logq
```

各入参含义：

- `w`：形状 `(B, 1, T)`，原始时长。
- `e_q`：形状 `(B, 2, T)`，正态噪声（官方是 `randn * x_mask`）。
- `z_u` / `z1`：形状各为 `(B, 1, T)`，`post_flows` 之后 `z_q` 拆出来的两半。
- `mask`：形状 `(B, 1, T)`。
- `post_logdets` / `flow_logdets`：各是长度 `B` 的列表，分别是 `post_flows` 与 `flows` 各自 logdet 之和。

返回：

```python
{
  "logq": [...], "nll": [...], "total": [...],
  "logdet_tot_q": [...], "logdet_tot": [...],
  "z0": [...], "z0_log": [...],
}
```

要点：

- ★★ `logq = gauss - logdet_tot_q`，而 `logdet_tot_q = post_logdets[b] + lsig`，其中 `lsig = sum((logsigmoid(z_u) + logsigmoid(-z_u)) * mask)`。
  **`lsig` 是被减去的**（它本身是负数，所以 `logq` 反而变大）。写成 `gauss + lsig - post` 就错了。
- `gauss = sum(-0.5 * (log(2*pi) + e_q**2) * mask)`，对 `e_q` 的两个 channel 与时间步求和。
- ★ `u = sigmoid(z_u) * mask`，`z0 = (w - u) * mask`：**`z0` 整体乘 mask**。
- `log_flow` 是 `Log` 基元：`z0_log = log(clamp_min(z0, 1e-5)) * mask`，`logdet_log = sum(-z0_log)`。
- `logdet_tot[b] = logdet_log[b] + flow_logdets[b]`。
- `z = cat([z0_log, z1])`，`nll = sum(0.5 * (log(2*pi) + z**2) * mask) - logdet_tot`，`total = nll + logq`。

## 常见错误清单

1. `Flip` 翻成时间轴（`torch.flip(x, [2])`）—— 官方是 `torch.flip(x, [1])`，channel 轴。
2. `Flip` 乘了 mask —— 官方 `Flip.forward` 根本没有 mask 参数。
3. `Log` 的 clamp 写成 `clamp_min(x, 1e-5)` 之后又乘 mask 的顺序反了（先乘 mask 再 clamp）。
4. `Log` 的反向忘了乘 mask —— 官方是 `torch.exp(x) * x_mask`。
5. `Log` 的 logdet 写成 `sum(y)` 而不是 `sum(-y)`。
6. `ElementwiseAffine` 的 logdet 忘了乘 mask（官方是 `sum(self.logs * x_mask, [1,2])`）。
7. `ElementwiseAffine` 的反向写成 `(x + m)` 而不是 `(x - m)`。
8. coupling 的前向把 mask 作用到整体（`(m + x1*exp(logs)) * mask`）—— 官方只乘在 `x1` 项上。
9. coupling 的 logdet 乘了 mask —— 官方 `sum(logs, [1,2])` 不乘 mask。
10. coupling 的 `x0` 也乘了 mask 或也做了变换 —— 官方 `x0` 原样透传。
11. coupling 的 `mean_only=True` 时仍用给定的 `logs` —— 官方此时 `logs = zeros_like(m)`。
12. block 的反向按正序执行 flow —— 官方是 `reversed(self.flows)`。
13. block 里把 `Flip` 放到了 coupling 之前 —— 官方顺序是 coupling 在前、flip 在后。
14. `logq` 的 `lsig` 项写成加号 —— 它是被减去的。
15. `z0` 忘了整体乘 mask。
16. `nll` 里 `z0_log` 与 `z1` 两半只算了一半。

## 自检

- `mode 1` 的 `flip`：给定 `x = [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]`，输出的前两个元素应是 `4.0` 与 `6.0`（channel 互换）。
- `mode 1` 的 `log` 反向：给定 `mask` 在某个位置为 0，那个位置的输出应是 `0`。
- `mode 2`：给定 `mask` 在某个 `x1` 位置为 0，那个位置的输出应是 `m`（而不是 `0`）。
- `mode 2`：mask 有 0 的位置，其 `logs` 仍然计入 logdet。
- `mode 3`：单个 coupling 的 block 反向，结果应是「先 flip 再 coupling 反向」的结果。
- `mode 4`：`logq` 一定大于 `gauss - post_logdets`（因为减掉了一个负数）。
