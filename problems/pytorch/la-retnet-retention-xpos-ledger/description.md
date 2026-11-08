# order 844 · RetNet 保留机制的三种表示等价账：指数衰减矩阵、递归状态递推与分块内跨块衰减

> 官方源：`Jamie-Stirling/RetNet`（RetNet 论文官方实现）——
> `src/retention.py`、`src/xpos_relative_position.py`

## 背景

RetNet（Retentive Network）用**固定指数衰减**代替 softmax 注意力做线性注意力。
它的核心机制叫 **retention**：序列第 `n` 个位置对第 `m` 个位置（`n >= m`）的影响
是 `γ^(n-m)` —— 一个与数据无关的**常数衰减核**。

同一套 retention 在官方代码里有**三种完全等价的表示**：

| 表示 | 复杂度 | 官方方法 |
|---|---|---|
| parallel | `O(L²)` | 直接算 `D ∘ (QK^T) V` |
| recurrent | `O(L)` | 递推状态 `s_n = γ s_{n-1} + K_n^T V_n` |
| chunkwise | `O(L·c)` | 块内用 D，跨块用递推状态 |

三种表示必须给出**同一个输出**。这是 RetNet 论文的核心断言，也是本题的核心。

RetNet 在 retention 之上还叠了一层 **xPos 相对位置编码**（`XPOS`）——
用一组可学习频率的 `base^(pos/512)` 对 query/key 做旋转缩放。
本题把 **retention 骨架（纯数值账）** 与 **xPos（含无理幂）** 严格隔离成四类 mode，
让判题强度最大化。

官方实现在 chunkwise 上有一个**真实的等价性 bug**（末块 xPos offset 错位），
本题显式复现它并把它与正确行为区分开。

---

## 口径（本题的唯一标准）

- **mode 1 / mode 2 是「纯 retention 骨架」**：不含 xPos。
  所有输入量是整数或有理数字符串（如 `"1/2"`、`"9/10"`），
  全链路用 `fractions.Fraction` **精确**计算，判定用**逐位相等**（不容差）。
- **mode 3 / mode 4 含无理量**（幂、三角函数、`exp`/`log`/`sqrt`）：
  - mode 3 的 `scale` / `sin` / `cos` 用**相对容差 `1e-9`**；
  - mode 4 的 `gammas` 用**相对容差 `1e-12`**，
    `y_cat` / `y_gn` / `out` 用**绝对容差**（`1e-8` / `1e-6` / `1e-6`），
    原因是 `GroupNorm` 的 `rstd = 1/sqrt(var + eps)` 会把上游 ~1e-11 的尾差
    放大到 1e-8 量级（详见 notes 的「尾差放大链」）。
- **`gamma` 必须在开区间 `(0, 1)` 内**（有理数字符串）。
  官方 `_get_D` 在 `n < m` 时算 `γ^(负数)`，可能 overflow 成 `inf`，
  `inf * 0 = NaN`，官方用 `D[D != D] = 0` 兜回 0。
  **本题只走 `n < m` 恒为 0 的语义，不引入 NaN 分支。**

---

## mode 1 · `decay_matrix`

官方 `SimpleRetention._get_D` 的衰减矩阵与 γ 的幂表。

**入参**

| 参数 | 类型 | 说明 |
|---|---|---|
| `gamma` | str | 有理数字符串，如 `"1/2"`、`"9/10"`、`"2/3"` |
| `seq_len` | int | 序列长度 `L`，`>= 1` |

**出参**

```json
{
  "gamma_powers": [...],   // γ^k, k = 0..seq_len-1
  "D": [[...]]             // L x L 的衰减矩阵
}
```

**语义**

- `gamma_powers[k] = γ^k`，`gamma_powers[0] = 1`。
- `D[n][m] = γ^(n-m)` 当 `n >= m`；**`n < m` 的位置恒为 `0`**。

**非法输入必须抛 `ValueError`**：`gamma` 不在 `(0, 1)`、`seq_len <= 0`。

---

## mode 2 · `retention_forms`

三种表示的等价账。**这是本题的核心。**

**入参**

`Q`、`K`、`V` 都是**按 head 分组的列表**：外层长度 = head 数 `H`，
每个元素是 `L` 行矩阵（`Q[h][n]` 是第 `h` 个 head、第 `n` 个位置的向量）。

| 参数 | 类型 | 说明 |
|---|---|---|
| `Q` | `[[[[int]]]]` | `H` 组，每组 `L` 行，行宽 = `head_dim` |
| `K` | `[[[[int]]]]` | 同上，行宽 = `head_dim` |
| `V` | `[[[[int]]]]` | 同上，**行宽 = `v_dim`（与 Q/K 的宽度独立）** |
| `gamma` | str | 有理数字符串，在 `(0, 1)` 内 |
| `chunk_size` | int | 块长 `c`，`>= 1`（默认 = `L`） |

**出参**

```json
{
  "parallel":    [[[...]]],   // (L, heads, v_dim)
  "recurrent":   [[[...]]],   // (L, heads, v_dim)
  "chunkwise":   [[[...]]],   // (L, heads, v_dim)
  "scores":      [[[...]]],   // (heads, L, L)
  "states":      [[[[...]]]], // (L, heads, head_dim, v_dim)
  "block_states": [[[[...]]]] // (块数, heads, head_dim, v_dim)
}
```

**三种表示的语义**

1. **parallel**：
   ```
   scores[h][n][m] = D[n][m] * sum_d Q[h][n][d] * K[h][m][d]
   parallel[n][h][e] = sum_m scores[h][n][m] * V[h][m][e]
   ```
   ⚠️ 输出是**逐 head 分开**的：`parallel[n][h]` 是第 `h` 个 head 在位置 `n` 的 `v_dim` 维向量。
   各 head 的 `v_dim` 相同（官方 `double_v_dim=False` 时 `v_dim = head_size`；
   本题允许 `V` 的宽度独立于 `Q`/`K` 的宽度）。

2. **recurrent**：
   ```
   s_n = γ * s_{n-1} + K_n^T V_n          （s_{-1} = 0）
   recurrent[n][h][e] = sum_d Q[h][n][d] * s_n[h][d][e]
   ```
   ⚠️ **递归形态下 `K_n`、`V_n` 只有当前一步**（形状 `(1, head_dim)` / `(1, v_dim)`），
   所以 `K_n^T V_n` 是**单步外积**：
   ```
   (K_n^T V_n)[d][e] = K[h][n][d] * V[h][n][e]
   ```
   **不做 `m` 求和。** 写成 `sum_m` 是最常见的错误。

   `states[n][h][d][e]` 就是 `s_n[h]` 的第 `(d, e)` 项。

3. **chunkwise**：把序列切成 `ceil(L / c)` 块，第 `bi` 块覆盖 `[bi*c, min((bi+1)*c, L))`，
   块内长度 `c' = min(c, L - bi*c)`。
   ```
   D_c        = 衰减矩阵（c' x c'，用同一个 γ）
   r_i        = (K_blk^T @ (V_blk * D_c[-1])) + (γ^c') * r_{i-1}
   inner      = ((Q_blk @ K_blk^T) * D_c) @ V_blk
   e[j]       = γ^(j+1)                              （j = 0..c'-1）
   cross      = (Q_blk @ r_{i-1}) * e
   输出        = inner + cross
   ```
   ⚠️ 四处细节：
   - **`D_c[-1]` 是衰减矩阵的最后一行** = `[γ^(c'-1), ..., γ^0]`，
     用来给 `V` 逐位置加权。
   - **`r_i` 的衰减是 `γ^(c')`**（`c'` = **本块实际长度**）。
     ⚠️ 官方 `forward_chunkwise` 内的局部变量 `chunk_size = x_i.shape[1]`
     就是**本块的实际长度**，所以末块（长度不足）时指数是 `γ^(c')` 而不是 `γ^c`。
     这个差异**在输出上不可观测**（末块状态 `r_i` 之后不再被使用），
     所以本题额外返回 `block_states`（每块结束时的 `r_i`）把它显式暴露出来判定。
   - **`cross` 用的是【旧状态 `r_{i-1}`】**，不是刚算出的 `r_i`。
     官方源码 `cross_chunk = (Q @ r_i_1) * e` 用的正是旧状态。
   - **`e[j] = γ^(j+1)`**（不是 `γ^j`）。

   `block_states[bi][h][d][e]` 就是第 `bi` 块结束时该 head 的状态 `r_i` 的第 `(d, e)` 项
   （官方 `forward_chunkwise` 也返回 `r_i`）。

   **本题口径**：chunkwise 的输出**必须与 parallel 逐位相等**（无论 `L` 是否被 `c` 整除）。
   原因见下面的「官方 chunkwise 的末块 quirk」。

**等价性断言（判题必查）**

```
parallel == recurrent == chunkwise        （逐位）
```

**官方 chunkwise 的末块 quirk**

官方 `forward_chunkwise(x_i, r_i_1, i)` 的第三个参数 `i` 是**块索引**，
函数内 `offset = i * chunk_size`，而 `chunk_size` 取的是 **`x_i.shape[1]`，即本块的实际长度**。

- 当 `L % c == 0` 时，每块长度都是 `c`，`offset = 块索引 * c` = 该块的起始绝对位置 ✓。
- **当 `L % c != 0` 时，末块长度 `c' < c`**，官方仍算 `offset = 块索引 * c'`，
  **不是真实的绝对起始位置**（真实位置是 `块索引 * c`）。

实测（L=7, c=3）：末块（块索引 2，长度 1）的 offset 变成 `2 * 1 = 2`，
而真实绝对位置是 `6` → **末块输出差 8.78e-02**（前 6 个位置差 ≤ 4.5e-8，纯浮点尾差）。

**受控实验（分离两个变量）**：

| 设定 | 与 parallel 的最大差 |
|---|---|
| 固定 xPos offset = 块起始绝对位置 | `4.47e-08`（纯尾差） |
| 固定 xPos offset 全为 0 | `5.07e-01`（语义错） |

⇒ **唯一根因是 xPos 的 offset**，与 `γ^c` / `γ^(c')` 无关。

**本题口径**：mode 2 是**纯骨架**（不含 xPos），offset 根本不进入计算，
所以 chunkwise 与 parallel **严格逐位相等**，无论 `L` 是否被 `c` 整除。

**非法输入必须抛 `ValueError`**：`Q`/`K`/`V` 为空、head 数不一致、
序列长度不一致、行宽为空、`gamma` 越界、`chunk_size <= 0`。

---

## mode 3 · `xpos_scale`

`XPOS` 相对位置的 scale / sin / cos 账。

**入参**

| 参数 | 类型 | 说明 |
|---|---|---|
| `head_dim` | int | 头维度，**必须是偶数**（`>= 2`） |
| `seq_len` | int | 序列长度 `L`，`>= 1` |
| `offset` | int | 位置偏移，`>= 0`（默认 0） |
| `downscale` | bool | 是否对 key 取倒数缩放（默认 `False`） |

**出参**

```json
{
  "base":          [...],    // head_dim / 2 个
  "scale":         [[...]],  // (L, head_dim/2)
  "sin":           [[...]],  // (L, head_dim/2)
  "cos":           [[...]],  // (L, head_dim/2)
  "applied_scale": [[...]]   // 实际乘到 sin/cos 上的 scale
}
```

**语义**

1. **base**：
   ```
   base[j] = (2j + 0.4 * head_dim) / (1.4 * head_dim)      j = 0..head_dim/2-1
   ```
   官方 `scale` buffer 是 `(torch.arange(0, head_dim, 2) + 0.4*head_dim) / (1.4*head_dim)`，
   **是 float32**（int tensor 与 Python float 相提升为 float32）。
   本题按 **float32 精度** 复刻这个 base。

2. **scale**：
   ```
   total = seq_len + offset
   scale[pos][j] = base[j] ** (pos / 512)         pos = 0..total-1
   ```
   `scale_base = 512`。

3. **截断**：`total > seq_len` 时（即 `offset > 0`），
   **`scale`、`sin`、`cos` 三者一起取最后 `seq_len` 行**。

4. **sin / cos**（`fixed_pos_embedding`）：
   ```
   dim = head_dim / 2                      ← 是 scale 的列数，不是 head_dim
   inv_freq[j] = 1 / (10000 ** (j / dim))
   angle[pos][j] = pos * inv_freq[j]
   sin[pos][j] = sin(angle[pos][j]) ;  cos 同理
   ```
   ⚠️ 官方 `inv_freq` 是 **float32**（`torch.arange(0, dim)` 无 dtype → float32），
   所以 `inv_freq[1] = 0.10000000149011612` 而不是精确的 `0.1`。
   本题按 float32 复刻 `inv_freq` 与角度乘积。

5. **官方顺序**：**先在完整的 `total` 行 `scale` 上算 sin/cos，再把三者一起截断**。
   先截断 scale 再算 sin/cos 是错的。

6. **downscale**：
   ```
   applied_scale = 1 / scale            （downscale=True 时）
   applied_scale = scale                （downscale=False 时）
   ```
   ⚠️ **取倒数发生在 `fixed_pos_embedding` 之后**，
   所以 **sin/cos 是用「未取倒数的 scale」算的**，downscale **不改变 sin/cos**。
   官方 `apply_rotary_pos_emb` 内部做 `duplicate_interleave(sin * scale)`，
   所以 `applied_scale` 才是真正乘到 sin/cos 上的那个 scale。

**非法输入必须抛 `ValueError`**：`head_dim` 为奇数或 `<= 0`、
`seq_len <= 0`、`offset < 0`。

---

## mode 4 · `multiscale_retention`

官方 `MultiScaleRetention` 的完整账：多尺度 γ + 逐 head retention + GroupNorm + swish 门控。

**入参**

| 参数 | 类型 | 说明 |
|---|---|---|
| `X` | `[[float]]` | `(L, hidden)` |
| `W_Q` / `W_K` | `[[[[float]]]]` | 每 head 一个 `(hidden, head_size)` 矩阵，共 `heads` 组 |
| `W_V` | `[[[[float]]]]` | 每 head 一个 `(hidden, head_size)` 矩阵，共 `heads` 组 |
| `W_G` | `[[float]]` | `(hidden, v_dim)`，门控投影 |
| `W_O` | `[[float]]` | `(v_dim, hidden)`，输出投影 |
| `heads` | int | head 数，`>= 1`；`hidden % heads == 0` |
| `chunk_size` | int | 块长（可选，默认 `L`） |

**出参**

```json
{
  "gammas": [...],    // heads 个
  "y_cat":  [[...]],  // (L, hidden)  逐 head 输出拼接
  "y_gn":   [[...]],  // (L, hidden)  GroupNorm 之后
  "out":    [[...]]   // (L, hidden)  最终输出
}
```

**语义**

1. **γ 序列**：
   ```
   γ_h = 1 - exp(linspace(log(1/32), log(1/512), heads)[h])
   ```
   首项精确为 `1 - 1/32 = 0.96875`，末项精确为 `1 - 1/512 = 0.998046875`，
   中间项含无理指数。`heads == 1` 时取 `1 - 1/32`。

2. **逐 head retention**：
   每个 head 用自己的 `γ_h` 跑一遍 parallel 形式的 retention，
   **并且必须带上 xPos**（官方 `SimpleRetention.forward` 里的
   `Q = self.xpos(Q)` 与 `K = self.xpos(K, downscale=True)`）。
   ⚠️ **xPos 是 retention 的必需组成部分**：实测官方 `SimpleRetention.forward`
   与「纯衰减骨架」的输出差 `0.0515`，加上 xPos 后差 `0.0`。

3. **cat**：`y_cat[n]` = 把 `heads` 个 head 的输出（每个 `head_size` 维）**首尾拼接**，
   总宽 `heads * head_size = hidden`（`v_dim = hidden`）。

4. **GroupNorm**（`nn.GroupNorm(heads, v_dim)`，`affine=True` → `weight=1`、`bias=0`，`eps = 1e-5`）：
   - 官方在 `Y.reshape(-1, v_dim)` 上做 → **对每个位置独立**（batch 维与 seq 维被拍平）。
   - 每组 `v_dim / heads` 个**连续通道**（按通道序切段，不是交错）。
   - **方差有偏**：`var = sum((x - mean)^2) / N`（分母是 `N`，不是 `N-1`）。
   - `y_gn = (x - mean) / sqrt(var + eps)`。
   - 归一化后的方差恒等于 `raw_var / (raw_var + eps)`。

5. **门控与输出**：
   ```
   out = (swish(X @ W_G) * y_gn) @ W_O
   swish(x) = x * sigmoid(x)
   ```

**非法输入必须抛 `ValueError`**：`X` 为空、`heads <= 0`、`hidden % heads != 0`、
`head_size` 为**奇数**（官方 `XPOS(head_size)` 的 `rotate_every_two` 在奇数 head_dim 时
`torch.stack` 尺寸不等直接崩，源码里「补零」的分支写在 `stack` 之后，是**死代码**）、
`W_Q`/`W_K`/`W_V` 的 head 数或行数不匹配。

---

## 常见错误清单

| 错误 | 后果 |
|---|---|
| `D[n][m]` 在 `n < m` 时算 `γ^(n-m)` | 负指数溢出成 `inf`/`NaN`，语义错 |
| 递归形态的 `K_n^T V_n` 写成 `sum_m` | 状态全错，输出全错 |
| `K_n^T V_n` 用整个序列的 `K`/`V` | 同上 |
| chunkwise 的 `cross` 用刚算出的 `r_i` 而非旧 `r_{i-1}` | 跨块贡献重复计入 |
| chunkwise 的 `r_i` 衰减用 `γ^c`（固定块长）而非 `γ^(c')`（官方语义：实际块长） | `block_states` 的末块错 |
| chunkwise 的 `e[j]` 写成 `γ^j` | 跨块衰减差一位 |
| chunkwise 的 `D_c[-1]` 取成第一行 | 块内加权反了 |
| `applied_scale` 取倒数后**又**重算 sin/cos | downscale 的语义错了 |
| sin/cos 用截断后的 scale 算 | 位置偏移全错 |
| `inv_freq` 用精确 `1/10000^(j/dim)` 而非 float32 | 尾差超出 `1e-9` |
| `fixed_pos_embedding` 的 `dim` 用 `head_dim` 而非 `head_dim/2` | sin/cos 全错 |
| `duplicate_interleave` 写成「每行复制一遍再拼接」 | 交错顺序错了 |
| `rotate_every_two` 写成「拼接」而非「交错」 | 旋转符号错 |
| 允许奇数 `head_dim` / `head_size` | 官方直接崩，必须报错 |
| 多尺度里漏掉 xPos | 输出差 `0.0515` |
| GroupNorm 用无偏方差（`N-1`） | `y_gn` 偏小 |
| GroupNorm 跨位置统计（对整行而非每位置） | `y_gn` 全错 |
| GroupNorm 按通道**交错**分组 | `y_gn` 全错 |
| `swish` 写成 `sigmoid(x)` | 门控错 |
| `gammas` 用 `linspace(1/32, 1/512)` 而非 `linspace(log(1/32), log(1/512))` | γ 序列全错 |
| mode 3/4 用精确有理数一路算到底 | 无理量被截断，容差不够 |

---

## 自检

```python
# mode 1：gamma = 1/2, L = 3
# gamma_powers = [1, 1/2, 1/4]
# D = [[1,   0,   0  ],
#      [1/2, 1,   0  ],
#      [1/4, 1/2, 1  ]]

# mode 2：单 head、L=2、gamma=1/2
# Q = [[1,2],[3,4]], K = [[1,0],[0,1]], V = [[1,0],[1,1]]
# scores[0][1][0] = 1/2 * 3 = 3/2 ;  scores[0][1][1] = 1 * 4 = 4
# parallel[1][0]   = (3/2)*[1,0] + 4*[1,1] = [11/2, 4]
# recurrent[1][0] 与 parallel[1][0] 逐位相等
# chunkwise[1][0] 与 parallel[1][0] 逐位相等（c=1 或 c=2 都是）
# block_states[0][0] = K_0^T V_0（单步外积），与 states[0][0] 相同（c=1 时）

# mode 3：head_dim=4
# base = [1.6/5.6, 3.6/5.6] = [2/7, 9/14]
# scale[0] 全为 1（base ** 0）
# sin[0] 全为 0 ; cos[0] 全为 1

# mode 4：GroupNorm 在 num_groups=1 时把整行归一化到零均值
# 归一化后的方差 = raw_var / (raw_var + 1e-5)
```
