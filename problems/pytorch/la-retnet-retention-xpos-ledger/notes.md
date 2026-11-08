# order 844 · 源码解剖：RetNet retention 的三种表示与 xPos

> 官方源：`Jamie-Stirling/RetNet`（GitHub 仓库 id `668291913`）
> 回源文件：`src/retention.py`（204 行）、`src/xpos_relative_position.py`（94 行）、
> `src/retnet.py`（76 行）

---

## 一、官方 retention 的三种表示

`SimpleRetention` 是 RetNet 的核心模块，一个 head 一个实例。
它有三个 `forward*` 方法，**必须给出同一个输出**（论文的核心断言）。

### 1.1 parallel 表示

```python
def forward(self, X):
    sequence_length = X.shape[1]
    D = self._get_D(sequence_length).to(self.W_Q.device)

    Q = (X @ self.W_Q)
    K = (X @ self.W_K)

    Q = self.xpos(Q)
    K = self.xpos(K, downscale=True)

    V = X @ self.W_V
    ret = (Q @ K.permute(0, 2, 1)) * D.unsqueeze(0)

    return ret @ V
```

形状流（`B` = batch，`H` = head，`L` = seq，`d` = head_size，`v` = v_dim）：

- `X`: `(B, L, hidden)`
- `Q`, `K`: `(B, L, d)`
- `V`: `(B, L, v)`
- `Q @ K.permute(0,2,1)`: `K.permute` 把 `(B, L, d)` 变成 `(B, d, L)`，
  广播成 `(B, L, L)` 的 `q_n · k_m`
- `D` 是 `(L, L)` → `D.unsqueeze(0)` 变 `(1, L, L)`，**逐 head 广播**
- `ret @ V`: `(B, L, L) @ (B, L, v)` → `(B, L, v)`

### 1.2 `_get_D` 的衰减矩阵与 NaN quirk

```python
def _get_D(self, sequence_length):
    n = torch.arange(sequence_length).unsqueeze(1)
    m = torch.arange(sequence_length).unsqueeze(0)

    # Broadcast self.gamma ** (n - m) with appropriate masking to set values where n < m to 0
    D = (self.gamma ** (n - m)) * (n >= m).float()  #this results in some NaN when n is much larger than m
    # fill the NaN with 0
    D[D != D] = 0

    return D
```

- `D[n][m] = γ^(n-m)`，当 `n >= m`；`n < m` 的位置被 `(n >= m)` 乘 0 抹掉。
- ⚠️ **官方 quirk**：`n < m` 时指数是**负数**，`γ^(-k)` 在数学上不等于 0，
  但被 `(n>=m)` 乘 0 抹掉；当 `γ` 是极小正数且 `n-m` 很负时，
  `γ^(n-m)` 可能 **overflow 成 `inf`**，`inf * 0 = NaN`，
  所以官方用 `D[D != D] = 0` 把 NaN 兜回 0。
  **本题只接受 `γ ∈ (0,1)` 且序列长度有限**，用**精确有理数**直接判定
  （`n < m` 位置恒为 0），**不走 NaN 分支**。
- 官方 `D` 的对角线恒为 `γ^0 = 1`。

### 1.3 recurrent 表示

```python
def forward_recurrent(self, x_n, s_n_1, n):
    """
    x_n: (batch_size, 1, hidden_size)
    s_n_1: (batch_size, hidden_size, v_dim)
    """
    Q = (x_n @ self.W_Q)
    K = (x_n @ self.W_K)

    Q = self.xpos(Q, n+1)
    K = self.xpos(K, n+1, downscale=True)

    V = x_n @ self.W_V

    s_n = self.gamma * s_n_1 + (K.transpose(-1, -2) @ V)

    return (Q @ s_n), s_n
```

- 状态 `s_n` 形状 `(B, head_size, v_dim)`（官方 docstring 写的 `(batch, hidden_size, v_dim)`，
  实际是 `head_size`）。
- ⚠️ **`xpos` 的 offset 是 `n+1`**（不是 `n`）—— 官方的一个 off-by-one 细节。
  这与 parallel 的 `xpos(Q)`（offset=0）不一致，所以官方 recurrent 与 parallel
  **在含 xPos 时并不严格相等**；本题 mode 2 是**纯骨架**（不含 xPos），offset 不进入计算。
- 衰减是 `γ * s_{n-1}`（每步乘一次 γ）。
- ★ **本题最容易踩的坑**：递归形态下 `x_n` 只有**当前一步**，
  所以 `K` 是 `(B, 1, d)`、`V` 是 `(B, 1, v)`，
  `K.transpose(-1,-2) @ V` 是 `(B, d, 1) @ (B, 1, v)` = **单步外积**，
  **不做 `m` 求和**。写成 `sum_m K[m][d] * V[m][e]` 是错的
  （我自己第一次实现就踩了这个，被对拍抓出来）。

### 1.4 chunkwise 表示

```python
def forward_chunkwise(self, x_i, r_i_1, i):
    """
    x_i: (batch_size, chunk_size, hidden_size)
    r_i_1: (batch_size, hidden_size, v_dim)
    """
    batch, chunk_size, _ = x_i.shape
    D = self._get_D(chunk_size)

    Q = (x_i @ self.W_Q)
    K = (x_i @ self.W_K)

    Q = self.xpos(Q, i * chunk_size)
    K = self.xpos(K, i * chunk_size, downscale=True)

    V = x_i @ self.W_V

    r_i =(K.transpose(-1, -2) @ (V * D[-1].view(1, chunk_size, 1))) + (self.gamma ** chunk_size) * r_i_1

    inner_chunk = ((Q @ K.transpose(-1, -2)) * D.unsqueeze(0)) @ V

    #e[i,j] = gamma ** (i+1)
    e = torch.zeros(batch, chunk_size, 1)

    for _i in range(chunk_size):
        e[:, _i, :] = self.gamma ** (_i + 1)

    cross_chunk = (Q @ r_i_1) * e

    return inner_chunk + cross_chunk, r_i
```

五处细节：

1. **`D[-1]` 是 `D` 的最后一行** = `[γ^(c-1), γ^(c-2), ..., γ^0]`（`c = chunk_size`），
   用来给 `V` 逐位置加权 → `K^T (V * D[-1])`。
2. **`r_i` 的衰减是 `γ^chunk_size`**，而官方函数内的 `chunk_size = x_i.shape[1]`
   是**本块的实际长度**（记作 `c'`），所以指数是 `γ^(c')` 而不是 `γ^c`（固定块长）。
   ⚠️ 这个差异在**输出上不可观测**（末块的 `r_i` 之后不再被使用），
   所以本题额外返回 `block_states`（每块结束时的 `r_i`）把它显式暴露出来判定。
3. **`e[j] = γ^(j+1)`**（注意是 `j+1` 不是 `j`），是跨块贡献的衰减因子。
4. **`inner_chunk`** 用的是**块内相对位置**的 `D`（`chunk_size × chunk_size`）。
5. ★ **`cross_chunk = (Q @ r_i_1) * e` 用的是【旧状态 `r_i_1`】**，
   **不是**同一函数里刚算出来的 `r_i`。
   我第一版实现写成了「用新算的 `r_prev`」，被官方对拍抓出（末块差 8.78e-02）。
   注意 `r_i` 的定义里已经含 `γ^chunk_size * r_i_1`，
   如果 cross 再用 `r_i`，跨块贡献就被重复计入且衰减了两次。

---

## 二、★★ 官方 chunkwise 与 parallel 不等价的真实 quirk

这是本题最核心的发现，也是官方实现里的一个**真实 bug**。

官方 `forward_chunkwise(x_i, r_i_1, i)` 的**第三个参数 `i` 是块索引**，函数内：

```python
Q = self.xpos(Q, i * chunk_size)     # chunk_size = x_i.shape[1]，是【本块实际长度】
```

- `retnet.py` 的调用是 `self.retentions[j].forward_chunkwise(x_i, r_i_1s[j], i)`，
  其中 `i` 是块索引，而 `x_i` 是**整块**（长度恒为 `chunk_size`）
  → `offset = 块索引 * 块长` = **该块的起始绝对位置** ✓。
- **但当序列长度不是块长的整数倍时，末块长度 `c' < chunk_size`**：
  官方仍用 `i * chunk_size`，而 `chunk_size` 取的是**末块的实际长度 `c'`**，
  于是 offset 变成 `块索引 * c'` —— **不是真实的绝对起始位置**。

实测（L=7, c=3）：末块（块索引 2，长度 1）的 offset 变成 `2 * 1 = 2`，
而真实绝对位置是 `6` → **末块输出差 8.78e-02**
（前 6 个位置差 ≤ 4.5e-8，纯浮点尾差）。

### 2.1 受控实验分离两个变量

我一开始怀疑是 `γ^chunk_size` vs `γ^(实际长度)` 的问题，
于是写了 `probe844d.py` 做受控实验，**把两个变量分开**：

| 设定 | 与 parallel 的最大差 |
|---|---|
| 固定 xPos offset = 块起始绝对位置 | `4.47e-08`（纯尾差） |
| 固定 xPos offset 全为 0 | `5.07e-01`（语义错） |

并且：**固定 offset 为绝对位置后，`γ**n`（实际长度）与 `γ**C`（固定块长）
两种衰减写法结果完全相同**。

⇒ **唯一根因是 xPos 的 offset**，与衰减因子无关。

### 2.2 本题口径

mode 2 是**纯 retention 骨架**（不含 xPos），offset 根本不进入计算，
所以 chunkwise 与 parallel **严格逐位相等**，无论 `L` 是否被 `c` 整除。
这样 mode 2 的判定是**最强的逐位判定**（不容差），把「骨架等价」与「位置编码」彻底隔离开。

---

## 三、xPos 相对位置（`src/xpos_relative_position.py`）

### 3.1 `XPOS` 主体

```python
class XPOS(nn.Module):
    def __init__(self, head_dim, scale_base=512):
        super().__init__()
        self.head_dim = head_dim
        self.scale_base = scale_base
        self.register_buffer(
            "scale", (torch.arange(0, head_dim, 2) + 0.4 * head_dim) / (1.4 * head_dim)
        )

    def forward(self, x, offset=0, downscale=False):
        length = x.shape[1]
        min_pos = 0
        max_pos = length + offset + min_pos
        scale = self.scale ** torch.arange(min_pos, max_pos, 1).to(self.scale).div(self.scale_base)[:, None]
        sin, cos = fixed_pos_embedding(scale)

        if scale.shape[0] > length:
            scale = scale[-length:]
            sin = sin[-length:]
            cos = cos[-length:]

        if downscale:
            scale = 1 / scale

        x = apply_rotary_pos_emb(x, sin, cos, scale)
        return x
```

十处要点：

1. **`scale_base = 512`**（默认）。
2. **`scale` buffer 长度是 `head_dim / 2`**（`arange(0, head_dim, 2)`）
   → **`head_dim` 必须是偶数**。
3. **`base[j] = (2j + 0.4*head_dim) / (1.4*head_dim)`**。
4. **`scale[pos][j] = base[j] ** (pos / 512)`**，`pos` 从 0 到 `length + offset - 1`。
5. **截断分支**：`scale.shape[0] = length + offset`；当 `> length` 时
   **取最后 `length` 行**（`scale[-length:]`），sin/cos 同截断。
   → **offset 越大，取的是越靠后的位置**。
6. ⚠️ **官方顺序**：**先在完整的 `total` 行 `scale` 上算 sin/cos，
   再把 scale / sin / cos 三者一起截断**。
   我最初写成「先截断 scale，再算 sin/cos」→ 位置全错。
7. ⚠️ **`downscale` 的确切位置**：`scale = 1/scale` 发生在
   **`fixed_pos_embedding(scale)` 之后**，所以 **sin/cos 是用「未取倒数的 scale」算的**，
   downscale **只把传给 `apply_rotary_pos_emb` 的 `scale` 参数取倒数**（它再乘在 sin/cos 上）。
   实测 downscale 与否输出差 `1.9e-02`（**不是恒等**）。
8. **`fixed_pos_embedding` 的 `dim` 是 `scale` 的列数** = `head_dim/2`，**不是 `head_dim`**。
9. **`duplicate_interleave`**：`(m0, m1, m2) → (m0, m0, m1, m1, m2, m2)`
   （每行元素各复制一次再交错）。
10. **`rotate_every_two`**：`x1 = x[..., ::2]`、`x2 = x[..., 1::2]`，
    `stack((-x2, x1), dim=-1).flatten(-2)` → `(a0, b0, a1, b1, ...)`（**交错**而非拼接）。

### 3.2 ★ 奇数 head_dim 官方直接崩

```python
def rotate_every_two(x):
    x1 = x[:, :, ::2]
    x2 = x[:, :, 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    if x.shape[-1]%2 == 1:
        # fill last dim with zero if hidden_size is odd
        x2 = torch.concat((x2, torch.zeros_like(x2[:, :, :1])), dim=-1)
    return x.flatten(-2)
```

`torch.stack((-x2, x1), dim=-1)` 要求两者等长，但偶数索引有 `ceil(d/2)` 个、
奇数索引有 `floor(d/2)` 个 → **`d` 为奇数时 `stack` 直接抛
`RuntimeError: stack expects each tensor to be equal size`**。

⚠️ 官方源码里「补零」的分支写在 `stack` **之后**，是**死代码**（永远到不了）。
实测 `head_dim = 5 / 7` → 崩；`4 / 6` 正常。

⇒ **本题要求 `head_dim`（mode 3）与 `head_size`（mode 4）必须是偶数**，奇数必须报错。

### 3.3 `fixed_pos_embedding` 的 float32 链

```python
def fixed_pos_embedding(x):
    seq_len, dim = x.shape
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim) / dim))
    sinusoid_inp = (
        torch.einsum("i , j -> i j", torch.arange(0, seq_len, dtype=torch.float), inv_freq).to(x)
    )
    return torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)
```

- `torch.arange(0, dim)` **无 dtype → float32** → `inv_freq` 全程 float32。
  所以 `inv_freq[1] = 0.10000000149011612`，**不是**精确的 `0.1`。
- `torch.arange(0, seq_len, dtype=torch.float)` 也是 float32；
  `einsum` 的乘积**也是 float32**，再 `.to(x)` 提升到 x 的 dtype（float64）。
- ⚠️ 所以 `angle[pos][j] = f32(f32(pos) * inv_freq[j])`，
  **不能**用 Python 的 `pos * inv_freq[j]`（double 乘法）。

### 3.4 `apply_rotary_pos_emb` 的乘法顺序

```python
def apply_rotary_pos_emb(x, sin, cos, scale=1):
    sin, cos = map(lambda t: duplicate_interleave(t * scale), (sin, cos))
    return (x * cos[:, :x.shape[-1]]) + (rotate_every_two(x) * sin)[:, :, :x.shape[-1]]
```

- `t * scale` 是**逐元素**乘（`t` 与 `scale` 同形 `(L, dim)`），
  然后 `duplicate_interleave` 把它交错复制到 `(L, 2*dim) = (L, head_dim)`。
- `cos[:, :x.shape[-1]]` 截断到 `head_dim`（当 `head_dim != 2*dim` 时才有区别，
  本题 `head_dim` 偶数时 `2*dim == head_dim`，恰好相等）。

---

## 四、`MultiScaleRetention`

```python
class MultiScaleRetention(nn.Module):
    def __init__(self, hidden_size, heads, double_v_dim=False):
        self.hidden_size = hidden_size
        self.v_dim = hidden_size * 2 if double_v_dim else hidden_size
        self.heads = heads
        assert hidden_size % heads == 0, "hidden_size must be divisible by heads"
        self.head_size = hidden_size // heads
        self.head_v_dim = hidden_size * 2 if double_v_dim else hidden_size

        self.gammas = (1 - torch.exp(torch.linspace(math.log(1/32), math.log(1/512), heads))).detach().cpu().tolist()

        self.swish = lambda x: x * torch.sigmoid(x)
        self.W_G = nn.Parameter(torch.randn(hidden_size, self.v_dim) / hidden_size)
        self.W_O = nn.Parameter(torch.randn(self.v_dim, hidden_size) / hidden_size)
        self.group_norm = nn.GroupNorm(heads, self.v_dim)

        self.retentions = nn.ModuleList([
            SimpleRetention(self.hidden_size, gamma, self.head_size, double_v_dim) for gamma in self.gammas
        ])

    def forward(self, X):
        Y = []
        for i in range(self.heads):
            Y.append(self.retentions[i](X))

        Y = torch.cat(Y, dim=2)
        Y_shape = Y.shape
        Y = self.group_norm(Y.reshape(-1, self.v_dim)).reshape(Y_shape)

        return (self.swish(X @ self.W_G) * Y) @ self.W_O
```

### 4.1 γ 序列

```
γ_h = 1 - exp(linspace(log(1/32), log(1/512), heads)[h])
```

- 首项精确 `1 - 1/32 = 0.96875`，末项精确 `1 - 1/512 = 0.998046875`。
- **中间项含无理指数**（`exp` of 非整 `log`）。
- `heads == 1` 时 `linspace` 只有一个点 = `log(1/32)` → γ = `0.96875`。
- ⚠️ 常见错误：写成 `linspace(1/32, 1/512, heads)`（**线性插值**而不是对数插值）→ 序列全错。

### 4.2 GroupNorm 的四个细节

`self.group_norm = nn.GroupNorm(heads, self.v_dim)`：

- `affine=True`（PyTorch 默认）→ 有可学习的 `weight` / `bias`，
  官方初始化时 `weight = 1`、`bias = 0` → 本题按恒等仿射处理。
- `eps = 1e-5`（PyTorch 默认）。
- 官方在 `Y.reshape(-1, self.v_dim)` 上做 → **对 (batch*seq) 每个位置独立**
  （batch 维与 seq 维一起被拍平成行维）。
- 每组 `v_dim / heads` 个**连续通道**（按通道序切段，**不是交错**）。
- ⚠️ **方差有偏**：`var = sum((x - mean)^2) / N`（分母 `N`，不是 `N-1`）。
- 归一化后该组的方差恒等于 `raw_var / (raw_var + eps)`。

### 4.3 输出形状流

```
X: (B, L, hidden)
  → 各 head: (B, L, head_size)          （含 xPos）
  → cat dim=2: (B, L, v_dim)            v_dim = heads * head_size = hidden
  → group_norm: (B, L, v_dim)
  → (swish(X @ W_G) * Y): (B, L, v_dim)
  → @ W_O: (B, L, hidden)
```

- `swish(x) = x * sigmoid(x)`。
- ★ **xPos 是 retention 的必需组成部分**：实测官方 `SimpleRetention.forward`
  与「纯衰减骨架」的输出差 `0.0515`；加上 xPos 后差 **0.0**。
  也就是说 mode 4 **不能**省掉 xPos。

---

## 五、尾差放大链（为什么 mode 4 的容差分层）

mode 4 里有一条**尾差放大链**，决定了容差必须分层：

```
xPos 的 base 是 float32          →  base 带 ~1e-8 相对尾差
  ↓ scale[pos][j] = base[j] ** (pos/512)
scale 有 ~1e-11 尾差
  ↓ Q @ K^T
scores 尾差 ~1e-11
  ↓ ret @ V（多项累加）
y_cat 尾差 ~1e-11
  ↓ GroupNorm: rstd = 1/sqrt(var + eps)，而 var ~ 8e-5
               → 相对误差被放大 1/sqrt(8e-5) ≈ 100 倍
y_gn 尾差 ~1e-8
  ↓ (swish(X @ W_G) * Y) @ W_O —— 16 项乘积的【灾难性抵消和】
   （项 ~0.03 而结果 ~0.0038，抵消掉 1 个数量级）
out 尾差再放大 1e3 倍 → ~5.7e-11 绝对
```

⇒ mode 4 的判定必须分层：

| 量 | 判定方式 |
|---|---|
| `gammas` | 相对容差 `1e-12` |
| `y_cat` | 绝对容差 `1e-8` |
| `y_gn` | 绝对容差 `1e-6` |
| `out` | 绝对容差 `1e-6` |

而 mode 1 / mode 2 是**纯有理数**，判定是**逐位相等**（不容差），
mode 3 的 `scale` / `sin` / `cos` 用**相对容差 `1e-9`**。

---

## 六、可复用的十条经验

1. **`_get_D` 的 NaN 分支不是装饰**：`γ^(负数)` 会 overflow 成 `inf`，
   `inf * 0 = NaN`。题目限定 `γ ∈ (0,1)` 后可以完全绕开，
   但要明白官方为什么写 `D[D != D] = 0`。
2. **递归形态的矩阵乘是单步外积**：`K_n`、`V_n` 只有当前一步，
   `K_n^T V_n` 不做 `m` 求和。这是最容易写错的一处。
3. **chunkwise 的 cross 用旧状态**：官方 `cross_chunk = (Q @ r_i_1) * e`，
   `r_i_1` 是**进函数时**的状态，不是刚算出的 `r_i`。
4. **衰减因子的指数是「本块实际长度」**：官方 `self.gamma ** chunk_size`
   里的 `chunk_size` 就是 `x_i.shape[1]`（本块实际长度 `c'`），不是固定块长。
   而这个差异**在输出上不可观测**，必须靠 `block_states` 才能判定。
5. **`e[j] = γ^(j+1)`**：跨块衰减因子带 +1 偏移。
6. **`D[-1]` 是最后一行**：给 `V` 逐位置加权用，取第一行会把加权反过来。
7. **xPos 的 sin/cos 在截断前算**：官方顺序是「完整行算 sin/cos → 三者一起截断」。
8. **downscale 在 `fixed_pos_embedding` 之后**：sin/cos 不变，只把乘到 sin/cos 上的 scale 取倒数。
9. **`fixed_pos_embedding` 的 `dim` 是 `head_dim/2`**：不是 `head_dim`。
   加上 `inv_freq` 全程 float32，尾差必须按 float32 复刻。
10. **官方死代码要当真的坑看**：`rotate_every_two` 的补零分支写在 `stack` 之后，
    奇数 `head_dim` 直接崩 —— 题目要求偶数并显式报错。

---

## 七、三层验证结构

本题采用**三层验证**：

### 第一层：官方源码回源

`src/retention.py` + `src/xpos_relative_position.py` 逐行读，
把每个数值细节（float32 精度、截断顺序、乘法的先后）都用探针脚本确认。

### 第二层：官方对拍（`/tmp/verify844.py`）

三层结构：
- **官方层**：复刻官方 `forward` / `forward_recurrent` / `forward_chunkwise`
  的**源码算式**（而不是直接调用类，因为官方 `SimpleRetention` 用同一个 `X`
  投影出 Q/K/V，无法注入互不相关的 Q/K/V）。
- **参考解层**：纯 `Fraction` 精确计算。
- **独立 oracle 层**：另写一份 Fraction 实现（不复用参考解代码），
  用来排除「参考解和 oracle 一起错」。

对拍结果（**四个 mode 全绿**）：

| mode | 最大相对误差 |
|---|---|
| mode 1（衰减矩阵） | `0.000e+00`（逐位精确） |
| mode 2（三表示） | `1.378e-14`（chunkwise 的纯浮点尾差；`parallel == recurrent` 逐位） |
| mode 3（xPos） | `1.723e-16` |
| mode 4（多尺度） | `4.985e-08`（xPos 浮点路径 + 灾难性抵消放大） |

### 第三层：mutation testing（`/tmp/mut844.py`）

对参考解做**单点变异**，每个变异都必须被判题用例抓住（`CAUGHT`）。
设计原则：**mutation 存活 = 判题缺口**，必须逐个补判题用例或修正变异。

★ **一个「不可观测差异」的发现**：mutation 报告显示，把 `γ^(c')` 改成 `γ^c`（固定块长）
**不会被输出抓住** —— 因为末块状态 `r_i` 之后不再被使用，
两种写法产生的最终输出**逐位相同**。
这暴露了一个判题缺口：**衰减指数在输出上不可观测**。
解决办法是按官方 `forward_chunkwise` 的返回值把每块状态 `r_i` 也暴露出来
（新增 `block_states` 输出），这样指数差异立刻可判定。
⇒ **mutation 存活不一定是「判题用例不够严」，也可能是「出参设计漏了一个可观测量」。**

本轮抓到的**真 bug**（都是变异存活后回查发现的）：

| 变异 | 后果 |
|---|---|
| `cross_chunk` 用新算的 `r_i` 而非旧 `r_i_1` | 末块差 `8.78e-02` |
| 递归的 `K_n^T V_n` 写成 `sum_m` | 状态全错 |
| `base` 公式漏掉分子的 `0.4*head_dim` | `2/7` vs `16/5` |
| `inv_freq` 用精确 `1/10000^(j/dim)` | `0.1` vs `0.10000000149011612` |
| sin/cos 用截断后的 scale 算 | 位置全错 |
| `scale` 全程 float32（官方 base 提升到 float64） | `0.9975562` vs `0.9976` |
| chunkwise 的 `r_i` 衰减用 `γ^c`（固定块长）而非官方的 `γ^(c')` | 输出相同（不可观测），但 `block_states` 的末块错 |

**真等价变异（必须剔除）**：
- `abs(w) if w > 0 else 0` 之类——它与 `max(w, 0)` 在 `w = 0` 时等价，
  是**真等价**，不是判题缺口。
