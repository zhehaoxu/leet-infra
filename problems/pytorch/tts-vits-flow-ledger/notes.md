# VITS 归一化流源码解剖

> 官方仓库：`jaywalnut310/vits`（`main` 分支）
> 涉及文件：`modules.py`、`models.py`
> 涉及类：`Log`、`Flip`、`ElementwiseAffine`、`ResidualCouplingLayer`、`ResidualCouplingBlock`、`StochasticDurationPredictor`

本文把官方实现逐条拆开，标出**不看源码就会写错**的地方，并说明本题的判题是怎么设计的。

## 一、官方 flow 的整体结构

VITS 的声学模型与时长预测器都用同一套「可逆变换 + log-det」的范式：

```
z ~ N(0, I)
  -> post_flows (coupling + flip 交替)      # 得到 z_q
  -> 拆成 z_u / z1
  -> u = sigmoid(z_u); z0 = (w - u)
  -> log_flow (Log 基元)
  -> flows (coupling + flip 交替)            # 得到 z
```

训练时的似然 = `nll + logq`，其中：

- `nll` 是**变换后**的 `z` 的标准正态负对数似然（减去 log-det）；
- `logq` 是**变换前**的 `z_q` 的负对数似然（减去 log-det）。

两条路径的 log-det 一正一负，合起来就是 flow 的「体积变化」。

## 二、★★ `Log`：clamp 与 logdet 的细节

官方 `modules.py`：

```python
class Log(nn.Module):
  def forward(self, x, x_mask, reverse=False, **kwargs):
    if not reverse:
      y = torch.log(torch.clamp_min(x, 1e-5)) * x_mask
      logdet = torch.sum(-y, [1, 2])
      return y, logdet
    else:
      x = torch.exp(x) * x_mask
      return x
```

三条要点：

1. **`clamp_min(x, 1e-5)` 在乘 mask 之前**。顺序不能换：先乘 mask 再 clamp 的话，mask 为 0 的位置会被 clamp 成 `1e-5` 从而得到 `log(1e-5) != 0`。
2. **`logdet = torch.sum(-y, [1, 2])`** —— 求和的两维是 channel 与 time（不是 batch），结果每 batch 一个。`y` 里已经含 mask，所以被 mask 的位置贡献 0。
3. ★ **反向分支是 `x = torch.exp(x) * x_mask`，也要乘 mask**。而且反向分支**只返回 `x`**，不返回 logdet（官方很多 flow 的 reverse 分支都这样）。

## 三、★★ `Flip`：翻的是 channel 轴

官方 `modules.py`：

```python
class Flip(nn.Module):
  def forward(self, x, *args, reverse=False, **kwargs):
    x = torch.flip(x, [1])
    if not reverse:
      logdet = torch.zeros(x.size(0)).to(dtype=x.dtype, device=x.device)
      return x, logdet
    else:
      return x
```

三条要点：

1. ★★ **`torch.flip(x, [1])` 翻的是第 1 维**。在 `(B, C, T)` 布局里第 1 维就是 **channel**。很多人在这里会写成 `[2]`（时间轴），实测形状 `(1, 2, 3)` 翻出来是 `(3, 2, 1)` 而不是 `(1, 2, 3)` 的时间反转。
2. ★ **签名是 `forward(self, x, *args, reverse=False, **kwargs)`** —— 根本没有 `x_mask` 这个形参，所以 **`Flip` 完全忽略 mask**。这不是「mask 被当成全 1」，而是「mask 压根没参与计算」。
3. `logdet` 恒为 0（翻转的行列式绝对值是 1，log 是 0）。反向分支也返回 `x`，且 forward 与 reverse 都是 flip（自逆）。

## 四、`ElementwiseAffine`：参数形状与整体 mask

官方 `modules.py`：

```python
class ElementwiseAffine(nn.Module):
  def __init__(self, channels):
    super().__init__()
    self.channels = channels
    self.m = nn.Parameter(torch.zeros(channels,1))
    self.logs = nn.Parameter(torch.zeros(channels,1))

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

三条要点：

1. ★ `m` / `logs` 是 **`nn.Parameter(torch.zeros(channels, 1))`**，形状是 `(channels, 1)` 而不是 `(channels,)`。本题把它放宽成「标量 / 长度 C 的一维列表 / `(C, 1)` 嵌套列表」三种写法。
2. ★ 前向是 **两行**：先 `y = self.m + torch.exp(self.logs) * x`，再 `y = y * x_mask`。**mask 作用在整体上**，mask 为 0 的位置输出 `0`（不是 `m`）。这跟 coupling 的行为不一样，见第五节。
3. `logdet = torch.sum(self.logs * x_mask, [1,2])` —— **logdet 要乘 mask**（跟 coupling 相反）。
4. 反向 `(x - self.m) * torch.exp(-self.logs) * x_mask`，mask 也在整体上。

## 五、★★ `ResidualCouplingLayer`：mask 只乘 x1 项

官方 `modules.py`：

```python
class ResidualCouplingLayer(nn.Module):
  def __init__(self, channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=0, mean_only=False):
    ...
    assert channels % 2 == 0
    ...

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

五条要点：

1. ★★ **前向 `x1 = m + x1 * torch.exp(logs) * x_mask`** —— mask **只乘在 `x1` 项上**，`m` 那一项**不被 mask**。所以 mask 为 0 的位置，输出是 **`m`** 而不是 `0`。这一点跟 `ElementwiseAffine` 的「整体 mask」截然不同，是本题最核心的判别点。
2. ★★ **`logdet = torch.sum(logs, [1, 2])` 没有乘 mask**。即使某些位置的 `x_mask` 是 0，那些位置的 `logs` 照样计入 logdet。（直觉上这有点怪 —— 但官方 `stats = self.post(h) * x_mask` 已经在上游把 mask 乘进 `logs` 了，所以这里「不乘 mask」是对的。本题把 `logs` 当作直接给定的量，于是就表现为「mask=0 的位置的 logs 也计入 logdet」。）
3. ★ `x0` **原样透传**，官方不对它做 mask 也不做变换。
4. `mean_only=True` 时 `logs = torch.zeros_like(m)`；`mean_only=False` 时 `m, logs = torch.split(stats, [half]*2, 1)`。官方 `ResidualCouplingBlock` 里**全部**用 `mean_only=True`，于是 block 的 logdet 恒为 0。
5. `assert channels % 2 == 0`；反向分支 `x1 = (x1 - m) * torch.exp(-logs) * x_mask`（mask 在整体上），且不返回 logdet。

### ★ 官方 `x_mask` 的形状不统一

这是本题最容易踩的坑：

| 类 | 需要的 mask 形状 |
| --- | --- |
| `Log` | 与 `x` 同形 `(B, C, T)` |
| `ElementwiseAffine` | 与 `x` 同形 `(B, C, T)` |
| `ResidualCouplingLayer` | `(B, 1, T)`，对 channel 广播 |
| `ResidualCouplingBlock` | `(B, 1, T)` |

官方 `models.py` 里 `ResidualCouplingBlock` 收的 `x_mask` 来自 `commons.sequence_mask`，形状是 `(B, 1, T)`。所以对拍时**必须准备两套 mask**：给 coupling/block 用 `(B, 1, T)`，给 `Log` / `ElementwiseAffine` 用展开后的 `(B, C, T)`。本题把两种形状都接受为入参，内部统一归一化到 `(B, C, T)`。

## 六、★★ `ResidualCouplingBlock`：正序 / 逆序

官方 `models.py`：

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

四条要点：

1. flow 序列固定为 `[Coupling_0, Flip, Coupling_1, Flip, ...]`，**coupling 在前、flip 在后**。
2. ★ **前向正序、反向逆序**（`reversed(self.flows)`）。coupling 与 flip **不满足交换律**，顺序换了结果就变。
3. 每个 coupling 都是 `mean_only=True` → `logs` 恒 0 → **block 的 logdet 恒为 0**。
4. 官方 forward 把 logdet 丢掉（`x, _ = flow(...)`），reverse 分支甚至只返回 `x`。

### ★★ 顺序差异在 n=2 时「恰好不可观测」

这是本题设计时踩到的一个非常隐蔽的坑。实测（`C=4, T=2`，mask 全 1）：

```
逆序 [F, C1, F, C0]  ->  [0.875, 1.25, 2.5, 4.25, 4.9, 5.8, 6.7, 7.6]
正序 [C0, F, C1, F]  ->  [0.875, 1.25, 2.5, 4.25, 4.9, 5.8, 6.7, 7.6]
```

**两者完全相同！** 原因：`mean_only=True` 时 coupling 退化成「第二半加一个常数」，flip 是 channel 反转 —— 在 `C=4`（两半各 2 个 channel）时这两步的组合对「正序 vs 逆序」恰好是对称的。

但 `n=1`（只有一个 coupling）时就能区分：

```
逆序 [F, C0]  ->  [3.0, 4.0, 0.5, 1.75]
正序 [C0, F]  ->  [2.5, 3.75, 1.0, 2.0]
```

⇒ **判题必须用单 coupling 的 block 来测 reverse 顺序**，用两个 coupling 是抓不住的。这是本轮 mutation 迭代中最有价值的一条发现。

## 七、★★ `StochasticDurationPredictor`：logq 的符号

官方 `models.py`：

```python
logdet_tot_q = 0
e_q = torch.randn([x.shape[0], 2, x.shape[2]]).to(device=x.device, dtype=x.dtype) * x_mask
z_q = e_q
for flow in self.post_flows:
  z_q, logdet_q = flow(z_q, x_mask, g=(x + h_w))
  logdet_tot_q += logdet_q

z_u, z1 = torch.split(z_q, [1, 1], 1)
u = torch.sigmoid(z_u) * x_mask
z0 = (w - u) * x_mask
logdet_tot_q += torch.sum((torch.logsigmoid(z_u) + torch.logsigmoid(-z_u)) * x_mask, [1, 2])
logq = torch.sum(-0.5 * (math.log(2 * math.pi) + (e_q ** 2)) * x_mask, [1, 2]) - logdet_tot_q

logdet_tot = 0
z0, logdet = self.log_flow(z0, x_mask)
logdet_tot += logdet
z = torch.cat([z0, z1], 1)
for flow in flows:
  z, logdet = flow(z, x_mask, g=x, reverse=reverse)
  logdet_tot = logdet_tot + logdet
nll = torch.sum(0.5 * (math.log(2 * math.pi) + (z ** 2)) * x_mask, [1, 2]) - logdet_tot
return nll + logq
```

六条要点：

1. ★★ **`lsig` 项是被减去的**。`logdet_tot_q = post_flows 的 logdet 之和 + lsig`，而 `logq = gauss - logdet_tot_q`。`lsig` 本身是负数（`logsigmoid(z) + logsigmoid(-z) <= -log 4`），所以减掉它会让 `logq` **变大**。写成 `gauss + lsig - post` 是最自然的一种错法，但符号完全反了。
2. ★ `u = torch.sigmoid(z_u) * x_mask`，`z0 = (w - u) * x_mask` —— **`z0` 整体乘 mask**。
3. `z0, logdet = self.log_flow(z0, x_mask)` —— `log_flow` 就是 `Log` 基元，`logdet` 是 `sum(-z0_log)`（负号在 `Log` 里面）。
4. `z = torch.cat([z0, z1], 1)` —— `z0`（已过 log）与 `z1`（没过 log）拼起来一起算高斯 NLL。
5. `nll` 与 `logq` 的高斯项符号相反：`nll` 是 `+0.5*(log(2pi)+z^2)`，`logq` 是 `-0.5*(log(2pi)+e_q^2)`。
6. `e_q = randn * x_mask` —— 噪声本身也乘 mask。

## 八、官方源码的三个「复现坑」

1. ★ **`modules.py` 只 `import scipy` 但不使用任何 scipy API**。官方 `modules.py` 顶部有 `import scipy.signal`，但整个文件没有一处调用 scipy。为了让官方模块能直接 import 对拍，在 `/tmp/vits846/` 放一个空的 `scipy.py` stub 就够了。
2. ★ **`models.py` 依赖 `attentions` 模块**（`from . import attentions`），而 `attentions` 又依赖 `torch.nn.functional` 之外的东西。整包 import 会失败。本题只需要 `modules.py` 的 flow 类，所以只对拍 `modules.py`。
3. ★ **`ElementwiseAffine` 的 `channels` 是硬编码在构造时的**。对拍脚本里如果写死 `ElementwiseAffine(1)`，就会在 `channels=2` 的用例上直接抛 shape 不匹配。要 `ElementwiseAffine(len(m))`。

## 九、判题设计的三层结构

| 层 | 角色 | 独立性 |
| --- | --- | --- |
| 官方层 | 直接 import `/tmp/vits846/modules.py`，用子类注入 `m` / `logs` 绕过 WN 网络 | 唯一真源 |
| 参考解层 | `/tmp/vits846_solution_staged.py` 的 `Fraction` 实现 | 独立重写 |
| oracle 层 | 判题用例里内联的 `_oracle_*` 函数 | 不依赖参考解 |

三层两两对拍：

- 官方 vs 参考解：22 个用例，最大差 `8.882e-16`（float64 的舍入量级）。
- oracle vs 参考解：最大差 `1.776e-15`。

oracle 层是判题里真正被断言的那一层，它**不 import 参考解**，所以参考解写错 oracle 不会跟着错。

## 十、★ 本轮新经验：等价变异要主动剔除

mutation 跑到第一轮时有 5 个存活，其中 **2 个是真等价变异**：

1. `cp-split-half-wrong`：我原本写的是在 `else` 分支里插入 `if c == half: pass` —— `pass` 什么也不做，**语义完全没变**。
2. `dur-z0-u-not-masked`：把 `u = _sigmoid(z_u) * mask` 改成 `u = _sigmoid(z_u)`。看起来是「忘了乘 mask」，但 `z0 = (w - u) * mask` 里 `u` 只出现在**整体乘 mask 之后**的位置 —— mask=0 时 `u` 无论取什么值，`z0` 都是 0。**数学上等价。**

处置：不是去补用例（补不出来），而是**换成别的变异**：

1. `cp-split-half-wrong` → `cp-split-halves-swapped`：把 `if c < half:` 改成 `if c >= half:`，让两半对调（真错）。
2. `dur-z0-u-not-masked` → `dur-u-sigmoid-removed`：把 `sigmoid(z_u)` 换成 `z_u`（忘了 sigmoid，真错）。

⇒ **★ 经验：mutation 存活时，第一件事是问「这是语义缺口还是等价变异」。** 等价变异要**主动删掉并替换**，不能留着当「已知存活」。

## 十一、★★ 本轮新经验：校验层要拆成互不依赖的「独立防线」

`logq` 的计算里有三处 mask 参与：`gauss`、`lsig`、`z0`。第一轮我只测了「`z0` 忘了乘 mask」，于是：

- `dur-gauss-mask-not-applied`（`gauss` 忘乘 mask）→ 存活
- `dur-lsig-mask-not-applied`（`lsig` 忘乘 mask）→ 存活
- `dur-z1-mask-not-applied`（`z1` 那半忘乘 mask）→ 存活

原因：**这三处共用了同一个测试数据的 mask 布局**，而测试数据的 mask 只在「batch 1 的 t=1」一处为 0 —— 只靠这一处，很多 mask 相关的错误都观察不到。

处置：把这三处拆成**三条互不依赖的防线**，每条一个独立变异，并各自补上**能被观察到的断言**：

```python
# 三条独立防线
gauss += -0.5 * (2.0 * half_log_2pi + float(eq[i]) ** 2) * float(mf[mi])
lsig  += (_log_sigmoid(zv) + _log_sigmoid(-zv)) * float(mf[mi])
acc   += 0.5 * (2.0 * half_log_2pi + zi1 ** 2) * float(mf[mi])
```

⇒ **★ 经验：同一语义不要在多层重复校验，否则 mutation 会存活。要拆成「每条防线只测一件事」的独立用例。**

## 十二、★★ 本轮新经验：不可观测的差异要设计专门的用例

三个存活项的根因各不相同：

| 存活变异 | 根因 | 处置 |
| --- | --- | --- |
| `log-reverse-exp-not-masked` | 用例缺口：reverse 的 Log 全用 mask=1 的数据 | 补 mask=0 的 reverse 用例 |
| `cp-logdet-masked` | 锚点缺陷：乘的是 `mf[C-1][0]`，那个值恰好恒为 1 | 重设锚点，乘「对应位置」的 mask |
| `blk-order-not-reversed` | n=2 时正序/逆序数值相同 | 补 n=1 的单 coupling 用例 |

其中 `cp-logdet-masked` 的教训最典型：我原本的变异是

```python
# 错的：乘的 mask 元素恰好恒为 1，等价于没改
acc += logslist[j] * mf[_index(b, C - 1, 0, B, C, T)]
```

正确的锚点要乘上「**该 `logs` 项自己对应位置的 mask**」：

```python
# 对的：按 (b, t) 定位到该 logs 项对应的 mask 元素
acc += logslist[j] * mf[_index(b, 0, (j - b * half * T) % T, B, C, T)]
```

⇒ **★ 经验：写 mutation 时，锚点必须落在「真正会变的那个输入」上。** 一个恒为 1 的 mask 元素会让整个变异退化成等价变异，而且**不会报错**——它只是静悄悄地存活。

## 十三、可复用的七条经验

1. ★★ `Flip` 在 `(B, C, T)` 布局里翻的是 **channel 轴**（`torch.flip(x, [1])`），而且**完全忽略 mask**。
2. ★★ coupling 的前向 mask **只乘 x1 项**，`m` 不被 mask → mask=0 处输出是 `m`；`ElementwiseAffine` 则是**整体 mask** → mask=0 处输出是 `0`。两者绝不能混。
3. ★★ coupling 的 `logdet` **不乘 mask**，`ElementwiseAffine` 的 logdet **乘 mask**。
4. ★★ `logq` 里 `logsigmoid` 那一项是**被减去的**（它本身是负数）。
5. ★ `logdet = sum(-y)` 的负号在 `Log` 里；`ElementwiseAffine` 是 `sum(logs * mask)`（正号）。
6. ★★ 官方 `x_mask` 的形状**不统一**：coupling/block 要 `(B, 1, T)`，`Log`/`ElementwiseAffine` 要 `(B, C, T)`。
7. ★★ block 的 reverse 是**逆序**执行，但这个差异在 **n=2 时不可观测**，必须用 **n=1** 的用例来测。

## 十四、三层验证结构（本轮的最终状态）

- 官方对拍：22 + 2 个用例，`official-vs-ref` 最大 `8.882e-16`，`oracle-vs-ref` 最大 `1.776e-15`。
- mutation：**39 项全部 CAUGHT，0 SURVIVED**（四轮迭代：28/36 → 36/36 → 39/39，含 3 处等价变异剔除与 3 处锚点重设）。
- 判题用例：4 个 `check_*`（`check_flow_primitive` / `check_coupling_layer` / `check_coupling_block` / `check_duration_nll`），每个都带内联 oracle。
