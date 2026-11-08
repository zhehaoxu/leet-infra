# Mamba2 SSD chunk 扫描三段账

## 背景

Mamba-2（`state-spaces/mamba`，`main` 分支）的 SSD（State Space Dual）把「选择性状态空间扫描」从
**逐 token 的串行递推**改写成**分块（chunk）并行**的三段流水线。官方在 Triton kernel 之外，
另提供了一份**纯 PyTorch 的参考实现**（`ssd_chunk_state.py::chunk_state_ref`、
`ssd_state_passing.py::state_passing_ref`、`ssd_chunk_scan.py::chunk_scan_ref`），
以及把三段串起来的 `ssd_combined.py::ssd_chunk_scan_combined_ref`。

这套流水线由三段组成，每段各管一件事：

1. **段一 `chunk_state_ref`**：算每个 chunk **内部**每个位置对 chunk 末状态的贡献，
   得到形状 `[b, nchunks, nheads, headdim, dstate]` 的状态张量。
2. **段二 `state_passing_ref`**：把各 chunk 的状态按**衰减加权**往后传，
   得到每个 chunk 的「传入前状态」与最终状态。
3. **段三 `chunk_scan_ref`**：算每个 chunk 内部的输出 =
   「chunk 内两两交叉项」+「初始状态项」，再加上 `D` 跳连与 `z` 门控。

本题要求你用**纯 Python**重建这三段账本，并处理官方实现里若干**不看源码就会写错**的细节。

## 口径约定

- 所有输入都是**嵌套列表**（不是展平的一维列表），形状必须严格匹配。
- 全程用 Python `float`（float64）计算。
- **符号与维度约定**（照抄官方）：

  | 符号 | 形状 | 含义 |
  |---|---|---|
  | `b` | int | batch |
  | `l` | int | seqlen |
  | `h` | int | nheads |
  | `p` | int | headdim |
  | `g` | int | ngroups（`h % g == 0` 必须成立） |
  | `n` | int | dstate |
  | `c` | int | nchunks = `l // chunk_size` |
  | `lc` | int | chunk_size |

- **`x`**：`[b][l][h][p]`。
- **`dt`**（时间步长）：`[b][l][h]`（原始布局）或 `[b][h][c][lc]`（官方 chunk 布局）。
  `chunk_state` / `chunk_scan` 要求**官方 chunk 布局** `[b][h][c][lc]`；
  `ssd_forward` 接受**原始布局** `[b][l][h]` 并自行完成 chunk 切分。
- **`A`**：`[h]`，每个 head 一个标量衰减系数（官方约定为负数）。
- **`B_mat` / `C_mat`**：`[b][l][g][n]`。
- **`D`**：`[h]` 或 `[h][p]`，或 `None`。
- **`z`**：`[b][l][h][p]`，或 `None`。
- **`dt_bias`**：`[h]`，或 `None`。
- **`chunk_size`**：正整数。
- **所有非法输入一律抛 `ValueError`**（错误消息要指明被违反的约束）。

## mode 1：`chunk_state` —— chunk 内状态账

```python
solve("chunk_state", x=..., dt=..., A=..., B_mat=..., chunk_size=...)
```

出参：

```json
{
  "dA":          [[[[...]]]],
  "dA_cumsum":   [[[[...]]]],
  "decay_states":[[[[...]]]],
  "expanded_B":  [[[[...]]]],
  "states":      [[[[[...]]]]],
  "nchunks": 2, "chunk_size": 4, "dstate": 4, "headdim": 3
}
```

- `dA[b][h][c][l]` = `dt[b][h][c][l] * A[h]`。
- `dA_cumsum[b][h][c][l]` = 该 chunk 内 `dA` 的前缀和（**chunk 内累积，不跨 chunk**）。
- `decay_states[b][h][c][l]` = `exp(dA_cumsum[b][h][c][lc-1] - dA_cumsum[b][h][c][l])`。
- `expanded_B[b][l][head][n]`：把 `B_mat` 的 group 维展开成 head 维。
- `states[b][c][h][p][n]`。

### ★ 要点 1.1：`decay_states` 是「相对 chunk 末尾」的衰减

官方的 `decay_states = torch.exp((dA_cumsum[:, :, :, -1:] - dA_cumsum))`，
减的方向是**末项减当前项**，所以：

- 最后一个位置的 `decay_states` **恒等于 `1.0`**（`exp(0)`）；
- 越靠前的位置，衰减越大（因为 `dA < 0` 时 `dA_cumsum` 单调下降，末项最小）。

**这与「从 chunk 开头累积」的衰减是相反的口径**，很容易写反。

### ★ 要点 1.2：group → head 的展开顺序是 `(g h)`，即 group 在前

官方写的是 `repeat(B, "b l g d -> b l (g h) d", h=nheads // ngroups)`。
`(g h)` 展平后，**`g` 是高位、`h` 是低位**，所以：

```
head 索引 = g * (h // G) + hi        (hi ∈ [0, h//G))
```

即 group `g` 对应 head 区间 `[g*(h//G), (g+1)*(h//G))`。
写成 `(h g)`（head 在前）时，在 `ngroups == 1` 的数据上**结果完全一样**，
只有多 group 的数据才能区分。

### ★ 要点 1.3：`states` 的 einsum 只对 chunk 内位置求和

官方写的是：

```python
torch.einsum("bclhn,bhcl,bhcl,bclhp->bchpn", B, decay_states, dt, x)
```

输出 `bchpn` 里含 `h` 与 `n`，输入里的 `h` 也出现在输出 ⇒ **`h` 不参与求和**，
只有 `l` 被求和。也就是说 `B` 与 `x` 必须用**同一个输出 head**，
不能对 head 再求一次和（多 head 时结果会不同）。

### ★ 要点 1.4：`dA_cumsum` 是 chunk 内累积，不跨 chunk

`torch.cumsum(dA, dim=-1)` 作用在 `[b][h][c][lc]` 的最后一维上，
所以**每个 chunk 都从零重新开始**。第 `c+1` 个 chunk 的首元素**不是**
第 `c` 个 chunk 末元素的延续。

## mode 2：`state_passing` —— chunk 间状态传递账

```python
solve("state_passing", states=..., dA_chunk_cumsum=..., initial_states=None)
```

出参：

```json
{
  "decay_chunk":  [[[[...]]]],
  "passed":       [[[[...]]]],
  "final_state":  [[[...]]],
  "nchunks": 2
}
```

- `states`：`[b][c][h][d]`，其中 `d = p * n`（`(headdim, dstate)` 已展平）。
- `dA_chunk_cumsum`：`[b][h][c]`，每个 chunk 末的 `dA_cumsum`。
- `initial_states`：`[b][h][d]` 或 `None`（`None` 时全零）。
- `decay_chunk`：`[b][h][c+1][c+1]`。
- `passed`：`[b][c][h][d]`（各 chunk 的「传入前状态」）。
- `final_state`：`[b][h][d]`。

### ★ 要点 2.1：`decay_chunk` 的尺寸是 `(c+1)` 而不是 `c`

官方先把初始状态拼到 states 前面（`cat` 后长度变 `c+1`），
再把 `dA_chunk_cumsum` 做 `F.pad(..., (1, 0))` 后 `cumsum`（长度也变 `c+1`），
然后构造 `(c+1) x (c+1)` 的下三角衰减矩阵。

所以 `decay_chunk` 的行列都是 `c+1`，**最后一行/最后一列对应「所有 chunk 之后」**。

### ★ 要点 2.2：矩阵方向不能反

官方的 `out = torch.einsum("bhzc,bchd->bzhd", decay_chunk, states)`：

- `decay_chunk[b][h][z][c]` —— **行 `z` 是输出 chunk，列 `c` 是被求和的 chunk**；
- `out[b][z][h][d] = Σ_c decay_chunk[b][h][z][c] * states[b][c][h][d]`。

转置这个矩阵会让结果不同（衰减会从「往后传」变成「往前传」）。

### ★ 要点 2.3：`decay_chunk` 是下三角（含对角），对角线恒为 `1.0`

`masked_fill(~tril(...), 0)` 只保留下三角（含对角）。
对角线是 `exp(0) = 1.0`。

### ★ 要点 2.4：`passed` 与 `final_state` 的切分

官方 `return out[:, :-1], out[:, -1]` —— `passed` 是**前 `c` 个**，
`final_state` 是**最后一个**。`passed` 的长度是 `c` 而不是 `c+1`。

## mode 3：`chunk_scan` —— chunk 内输出账

```python
solve("chunk_scan", B_mat=..., C_mat=..., x=..., dt=..., dA_cumsum=...,
      prev_states=..., D=None, z=None)
```

出参：

```json
{
  "CB":           [[[[[...]]]]],
  "decay":        [[[[[...]]]]],
  "scores_decay": [[[[[...]]]]],
  "out":          [[[[...]]]]
}
```

- `dt`：`[b][h][c][lc]`；`dA_cumsum`：`[b][h][c][lc]`。
- `prev_states`：`[b][c][h][p][n]`。
- `CB[b][c][h][l][s]`、`decay[b][h][c][l][s]`、`scores_decay[b][c][h][l][s]`。
- `out[b][l][h][p]`。

### ★ 要点 3.1：`CB` 是 chunk 内的两两交叉项

官方：

```python
CB = torch.einsum("bclhn,bcshn->bchls",
                   rearrange(C, "b (c l) h n -> b c l h n", c=nchunks),
                   rearrange(B, "b (c s) h n -> b c s h n", c=nchunks))
```

即 `CB[b][c][h][l][s] = Σ_n C[b][c*lc+l][h][n] * B[b][c*lc+s][h][n]`。
`C` 与 `B` 都**先经过 group→head 展开**再参与计算。

### ★ 要点 3.2：`decay[l][s]` 的方向是「当前减参考」

官方 `dt_segment_sum = dA_cumsum[:,:,:,:,None] - dA_cumsum[:,:,:,None,:]`，
所以 `decay[b][h][c][l][s] = exp(dA_cumsum[l] - dA_cumsum[s])`。

`dA < 0` ⇒ `l >= s` 时 `dA_cumsum[l] <= dA_cumsum[s]` ⇒ `decay <= 1`（衰减）；
`l < s` 时指数为正 ⇒ `decay > 1`，但随后被下三角 mask 清零 ⇒ **因果**。

### ★ 要点 3.3：`scores_decay` 是下三角（含对角）

`masked_fill(~tril(chunk_size, chunk_size), 0)` ⇒ `s > l` 的位置为 `0`。

### ★ 要点 3.4：输出是「交叉项 + 初始状态项」

```python
out = torch.einsum('bchls,bhcs,bcshp->bclhp', scores_decay, dt, x)
state_decay_out = torch.exp(rearrange(dA_cumsum, "b h c l -> b c l h 1"))
out_prev = torch.einsum('bclhn,bchpn->bclhp', C, prev_states) * state_decay_out
out = out + out_prev
```

注意 `state_decay_out` 是 **chunk 内逐位置**的 `exp(dA_cumsum[l])`，
不是 chunk 末的累积值。

### ★ 要点 3.5：`D` 是**加性**跳连，`z` 是**乘性**门控

```python
if D is not None:
    if D.dim() == 1:
        D = rearrange(D, "h -> h 1")
    out = out + x * D
return out if z is None else out * F.silu(z)
```

- 一维 `D`（`[h]`）被 reshape 成 `[h][1]`，广播到 `headdim`；
- `z` 是逐元素的 `silu` 门控（**乘**），不是加；
- `silu(z) = z * sigmoid(z)`，**不是** `sigmoid(z)` 本身。

## mode 4：`ssd_forward` —— 三段完整编排

```python
solve("ssd_forward", x=..., dt=..., A=..., B_mat=..., C_mat=..., chunk_size=...,
      D=None, z=None, dt_bias=None, dt_softplus=False)
```

- `dt`：**原始布局** `[b][l][h]`。
- `dt_softplus`：布尔，默认 `False`。

出参：

```json
{
  "out":            [[[[...]]]],
  "dA_cumsum":      [[[[...]]]],
  "decay_states":   [[[[...]]]],
  "states":         [[[[[...]]]]],
  "prev_states":    [[[[[...]]]]],
  "decay_chunk":    [[[[...]]]],
  "final_state":    [[[...]]],
  "CB":             [[[[[...]]]]],
  "scores_decay":   [[[[[...]]]]],
  "nchunks": 2, "chunk_size": 4, "padded_seqlen": 8
}
```

编排顺序照抄官方 `ssd_chunk_scan_combined_ref`：

```python
dt = F.pad(dt, (0, 0, 0, chunk_size - seqlen % chunk_size))   # 补齐到 chunk 整数倍
dt = rearrange(dt, "b (c l) h -> b h c l", l=chunk_size)
dt = dt.float()
if dt_bias is not None:
    dt = dt + rearrange(dt_bias, "h -> h 1 1")
if dt_softplus:
    dt = F.softplus(dt)
dA = dt * rearrange(A, "h -> h 1 1")
dA_cumsum = torch.cumsum(dA, dim=-1)
states = chunk_state_ref(B, x, dt, dA_cumsum)
states = state_passing_ref(rearrange(states, "... p n -> ... (p n)"),
                           dA_cumsum[:, :, :, -1])[0]
states = rearrange(states, "... (p n) -> ... p n", n=dstate)
out = chunk_scan_ref(B, C, x, dt, dA_cumsum, states, D=D, z=z)
```

### ★ 要点 4.1：`dt_bias` 在 `softplus` **之前**相加

官方顺序是「先加 bias，再 softplus」。所以 `dt_bias` 的影响不是
`softplus(dt + bias)`，而是 `softplus(dt + bias)` 之后**再乘 `A`**。

### ★ 要点 4.2：`dt_bias` 逐 head 独立

`rearrange(dt_bias, "h -> h 1 1")` ⇒ 每个 head 一个标量，
广播到该 head 的**所有 chunk、所有位置**。

### ★ 要点 4.3：`(headdim, dstate)` 的展平是 C 序

官方 `rearrange(states, "... p n -> ... (p n)")` 与还原
`rearrange(states, "... (p n) -> ... p n", n=dstate)` 是一对，
展平索引为 `p * n + ni`（`p` 是高位）。

### ★ 要点 4.4：★ 官方参考实现自身不自洽 —— `seqlen` 非整数倍会崩

官方 `ssd_chunk_scan_combined_ref` 把 `dt` 补齐到 `chunk_size` 的整数倍，
但下游 `chunk_scan_ref` 里有一句：

```python
assert seqlen == nchunks * chunk_size
```

用的是**原始 `seqlen`** ⇒ 当 `seqlen % chunk_size != 0` 时**直接抛 AssertionError**。

也就是说官方参考实现在这条路径上**崩掉**（Triton kernel 走的是另一条路径，不受影响），
而 `chunk_state_ref` 里的 `F.pad` 补齐分支在这条路径上**永远走不到**。

**本题要求在这一情形显式抛 `ValueError`**（错误消息里要能看出是 `ssd_forward` 自己拒绝的）。

## 常见错误清单

1. `decay_states` 写成「从 chunk 开头累积」的衰减（方向反了）。
2. `decay_states` 的最后一项不保证是 `1.0`（忘了 `exp(0)`）。
3. `dA_cumsum` 跨 chunk 累积（官方是 chunk 内累积）。
4. group→head 展开写成 `(h g)` 顺序（`ngroups == 1` 时看不出问题）。
5. `states` 的 einsum 对 head 多求了一次和。
6. `decay_chunk` 的尺寸写成 `c x c`（官方是 `(c+1) x (c+1)`）。
7. `decay_chunk` 转置（行是输出 chunk，列是被求和的 chunk）。
8. `decay_chunk` 用上三角而不是下三角。
9. `passed` 的长度写成 `c+1`（官方是 `c`，最后一项单独作为 `final_state`）。
10. `state_passing` 忘了把初始状态拼在前面（`cat` 后长度变 `c+1`）。
11. `CB` 用的是**未展开**的 `C` / `B`（必须先 group→head 展开）。
12. `decay[l][s]` 的方向写成「参考减当前」。
13. `scores_decay` 忘了下三角 mask（未来位置泄漏）。
14. `state_decay_out` 用了 chunk 末的累积值（应是 chunk 内逐位置值）。
15. `D` 当成乘法（官方是 `out + x * D` 的**加性**跳连）。
16. `z` 当成加法（官方是 `out * silu(z)` 的**乘性**门控）。
17. `silu(z)` 写成 `sigmoid(z)`。
18. `dt_bias` 加在 `softplus` 之后。
19. `dt_bias` 被当成逐位置量（官方是逐 head 一个标量）。
20. `(headdim, dstate)` 展平成 `(dstate, headdim)` 顺序。
21. `ssd_forward` 在 `seqlen` 非整数倍时静默补齐（官方是崩，本题要求报错）。
22. `seqlen` 非整数倍时 `chunk_size` 的 `F.pad` 分支被误当成「生效」。

## 自检

```python
solve("chunk_state", x=..., dt=..., A=..., B_mat=..., chunk_size=4)
```

应能拿到 `decay_states` 每行最后一项恰为 `1.0`、`dA_cumsum` 每个 chunk 从零开始、
`expanded_B` 的 head 索引满足 `head = g * (h // G) + hi`。

```python
solve("state_passing", states=..., dA_chunk_cumsum=...)
```

应能拿到 `decay_chunk` 的尺寸是 `(c+1) x (c+1)`、对角线全为 `1.0`、
`len(passed) == c`。

```python
solve("chunk_scan", B_mat=..., C_mat=..., x=..., dt=..., dA_cumsum=...,
      prev_states=..., D=[...], z=...)
```

应能拿到 `scores_decay` 的上三角全为 `0`、把 `z` 置 `None` 后 `out` 变化
（乘性门控）、把 `D` 置 `None` 后 `out` 变化（加性跳连）。

```python
solve("ssd_forward", x=..., dt=..., A=..., B_mat=..., C_mat=..., chunk_size=4)
```

三段的中间量（`states` / `prev_states` / `final_state`）应与
分别调用 `chunk_state` / `state_passing` / `chunk_scan` 的结果逐位一致。
