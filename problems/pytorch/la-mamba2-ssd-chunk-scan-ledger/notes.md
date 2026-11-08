# Mamba2 源码解剖：SSD chunk 扫描三段账

> 官方仓库：`state-spaces/mamba`（分支 `main`）
> 涉及文件：`mamba_ssm/ops/triton/ssd_chunk_state.py`、`mamba_ssm/ops/triton/ssd_state_passing.py`、
> `mamba_ssm/ops/triton/ssd_chunk_scan.py`、`mamba_ssm/ops/triton/ssd_combined.py`

## 1. 这套规则在 Mamba 里的位置

Mamba-1（`la-mamba-selective-scan` 那题）把状态空间扫描写成**逐 token 的串行递推**：
每个时间步做一次 `h = A h + B x`，再乘 `C` 取输出。串行依赖让 GPU 吃不满。

Mamba-2 的 SSD（State Space Dual）把同一件事改写成**分块并行**：
把 `l` 个 token 切成 `c = l // chunk_size` 个 chunk，每个 chunk 内部用矩阵乘并行算，
chunk 之间只传**状态**。官方在 Triton kernel 之外另给了一份**纯 PyTorch 参考实现**，
就是本题的对象。

三段分工：

| 段 | 官方函数 | 管什么 |
|---|---|---|
| 一 | `chunk_state_ref` | chunk 内每个位置对 chunk 末状态的贡献 |
| 二 | `state_passing_ref` | chunk 之间状态的衰减加权传递 |
| 三 | `chunk_scan_ref` | chunk 内输出（交叉项 + 初始状态项 + D 跳连 + z 门控） |

```
dA = dt * A ;  dA_cumsum = cumsum(dA)          （chunk 内累积）
        │
        ├─ 段一：states[b][c][h][p][n]
        │        = Σ_l B · decay_states · dt · x
        │
        ├─ 段二：decay_chunk（c+1 × c+1 下三角）
        │        passed[b][c][h][p*n] / final_state[b][h][p*n]
        │
        └─ 段三：CB → scores_decay → out
```

## 2. 段一 `chunk_state_ref` 的六个要点

官方代码（`ssd_chunk_state.py:1094-1122`）：

```python
def chunk_state_ref(B, x, dt, dA_cumsum):
    batch, seqlen, nheads, headdim = x.shape
    dstate = B.shape[-1]
    _, _, nchunks, chunk_size = dt.shape
    assert seqlen <= nchunks * chunk_size
    ngroups = B.shape[2]
    assert nheads % ngroups == 0
    B = repeat(B, "b l g d -> b l (g h) d", h=nheads // ngroups)
    if seqlen < nchunks * chunk_size:
        x = F.pad(x, (0, 0, 0, 0, 0, nchunks * chunk_size - seqlen))
        B = F.pad(B, (0, 0, 0, 0, 0, nchunks * chunk_size - seqlen))
    x = rearrange(x, "b (c l) h p -> b c l h p", l=chunk_size)
    B = rearrange(B, "b (c l) ... -> b c l ...", l=chunk_size)
    decay_states = torch.exp((dA_cumsum[:, :, :, -1:] - dA_cumsum))
    return torch.einsum("bclhn,bhcl,bhcl,bclhp->bchpn", B.to(x.dtype),
                         decay_states.to(x.dtype), dt.to(x.dtype), x)
```

### ★ 要点 2.1：`decay_states` 的减法方向是「末项减当前项」

`dA_cumsum[:, :, :, -1:] - dA_cumsum` ⇒ 每一项是
`exp(dA_cumsum[lc-1] - dA_cumsum[l])`。

实测（M=1, h=1, chunk=4）：`[0.716424, 0.764340, 0.856528, 1.0]` ——
最后一项恰好 `1.0`，且**越靠前越小**（衰减更强）。

写成 `dA_cumsum[l] - dA_cumsum[lc-1]` 就把方向搞反了：那样得到的是
`[1.39, 1.31, 1.17, 1.0]`，**大于 1**，是「放大」而不是「衰减」。

### ★ 要点 2.2：`(g h)` 展平是 group 在前

`repeat(B, "b l g d -> b l (g h) d", h=nheads // ngroups)` 展开后
`head = g * (h // G) + hi`。

★ **在 `ngroups == 1` 的数据上，`(g h)` 与 `(h g)` 两种顺序完全一样** ——
这是本题最容易漏的判题陷阱：必须用多 group 的数据才能区分。

### ★ 要点 2.3：einsum 只对 `l` 求和

`torch.einsum("bclhn,bhcl,bhcl,bclhp->bchpn", ...)`：
输出 `bchpn` 含 `h`、`n`，输入里的 `h` 也在输出里 ⇒ `h` **不参与求和**。
只有 `l` 是求和索引。

所以 `B5[b][c][l][h][n]` 与 `x5[b][c][l][h][p]` 必须用**同一个** `h`，
不能写成 `Σ_{l,h'}`。多 head 时这两者结果不同。

### ★ 要点 2.4：`dA_cumsum` 是 chunk 内累积

`torch.cumsum(dA, dim=-1)` 作用在 `[b][h][c][lc]` 的最后一维 ⇒ **每个 chunk 从零开始**。

实测：chunk 0 末 = `-0.489734`，chunk 1 首 = `-0.147134` ——
**不是**接着 chunk 0 继续累加。

### ★ 要点 2.5：`seqlen` 的 `F.pad` 分支与下游断言矛盾

`chunk_state_ref` 里有 `if seqlen < nchunks * chunk_size: F.pad(...)`，
但 `chunk_scan_ref` 里有 `assert seqlen == nchunks * chunk_size`，
而 `ssd_chunk_scan_combined_ref` 里 `dt` 被 pad 到整数倍、`seqlen` 却还是原值。

⇒ **在这条官方参考路径上，`seqlen % chunk_size != 0` 会直接 AssertionError 崩掉**，
`chunk_state_ref` 的 F.pad 分支**永远走不到**（只有单独调 `chunk_state_ref` 时才生效）。

### ★ 要点 2.6：`nheads % ngroups == 0` 是硬约束

官方 `assert nheads % ngroups == 0`。因为 `h // G` 是整除，
若不整除则展开后的 head 数与 `nheads` 不符，后续索引会错位。

## 3. 段二 `state_passing_ref` 的四个要点

官方代码（`ssd_state_passing.py:327-350`）：

```python
def state_passing_ref(states, dA_chunk_cumsum, initial_states=None):
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, 0])
    states = torch.cat([rearrange(initial_states, "b h d -> b 1 h d"), states], dim=1)
    dA_chunk_cumsum = F.pad(dA_chunk_cumsum, (1, 0))
    dA_chunk_cumsum = torch.cumsum(dA_chunk_cumsum, dim=-1)
    nchunks = dA_chunk_cumsum.shape[-1]
    dt_chunk_segment_sum = dA_chunk_cumsum[:, :, :, None] - dA_chunk_cumsum[:, :, None, :]
    decay_chunk = torch.exp(dt_chunk_segment_sum)
    causal_mask = torch.tril(torch.ones(nchunks, nchunks, device=states.device, dtype=bool), diagonal=0)
    decay_chunk = decay_chunk.masked_fill(~causal_mask, 0)
    out = torch.einsum("bhzc,bchd->bzhd", decay_chunk.to(dtype=states.dtype), states)
    return out[:, :-1], out[:, -1]
```

### ★ 要点 3.1：两处长度都从 `c` 变成 `c+1`

- `cat` 把初始状态拼在前面 ⇒ `states` 长度 `c+1`；
- `F.pad(dA_chunk_cumsum, (1, 0))` + `cumsum` ⇒ `dA_chunk_cumsum` 长度 `c+1`。

所以 `decay_chunk` 是 **`(c+1) x (c+1)`**，且 `nchunks = dA_chunk_cumsum.shape[-1]`
取的是**补齐后**的长度（实测 M=1/2 chunk 时是 3×3 而不是 2×2）。

### ★ 要点 3.2：矩阵方向不能反

`einsum("bhzc,bchd->bzhd")`：`decay_chunk[b][h][z][c]`，
**行 `z` 是输出 chunk、列 `c` 是被求和的 chunk**。

实测（c=2）的 `decay_chunk[0][0]`：

```
[1.0,      0.0,      0.0     ]
[0.61279,  1.0,      0.0     ]
[0.398209, 0.649829, 1.0     ]
```

第 0 行全 0 在对角线之外 ⇒ 初始状态那一列只被对角线用到。

### ★ 要点 3.3：`out[:, :-1]` 与 `out[:, -1]` 的切分

`passed` 是**前 `c` 个**，`final_state` 是**最后一个**。
`passed` 的长度是 `c`，不是 `c+1`。

### ★ 要点 3.4：初始状态默认是零

`torch.zeros_like(states[:, 0])` —— 形状是 `[b][h][d]`，不是 `[b][c][h][d]`。

## 4. 段三 `chunk_scan_ref` 的五个要点

官方代码（`ssd_chunk_scan.py:1846-1887`）：

```python
def chunk_scan_ref(B, C, x, dt, dA_cumsum, prev_states, D=None, z=None):
    batch, seqlen, nheads, headdim = x.shape
    _, _, ngroups, dstate = B.shape
    assert B.shape == (batch, seqlen, ngroups, dstate)
    _, _, nchunks, chunk_size = dt.shape
    assert seqlen == nchunks * chunk_size
    assert C.shape == B.shape
    B = repeat(B, "b l g d -> b l (g h) d", h=nheads // ngroups)
    C = repeat(C, "b l g d -> b l (g h) d", h=nheads // ngroups)
    CB = torch.einsum("bclhn,bcshn->bchls", rearrange(C, "b (c l) h n -> b c l h n", c=nchunks),
                      rearrange(B, "b (c s) h n -> b c s h n", c=nchunks))
    dt_segment_sum = dA_cumsum[:, :, :, :, None] - dA_cumsum[:, :, :, None, :]
    decay = torch.exp(dt_segment_sum)
    scores_decay = CB * rearrange(decay, "b h c l s -> b c h l s")
    causal_mask = torch.tril(torch.ones(chunk_size, chunk_size, device=x.device, dtype=bool), diagonal=0)
    scores_decay = scores_decay.masked_fill(~causal_mask, 0)
    out = torch.einsum('bchls,bhcs,bcshp->bclhp', scores_decay.to(x.dtype), dt.to(x.dtype),
                       rearrange(x, "b (c s) h p -> b c s h p", c=nchunks))
    state_decay_out = torch.exp(rearrange(dA_cumsum, "b h c l -> b c l h 1"))
    out_prev = torch.einsum('bclhn,bchpn->bclhp', rearrange(C, "b (c l) h n -> b c l h n", c=nchunks),
                            prev_states.to(C.dtype)) * state_decay_out
    out = out + out_prev
    out = rearrange(out, "b c l h p -> b (c l) h p")
    if D is not None:
        if D.dim() == 1:
            D = rearrange(D, "h -> h 1")
        out = out + x * D
    return out if z is None else out * F.silu(z)
```

### ★ 要点 4.1：`CB` 用的是**展开后**的 `C` 与 `B`

`B` / `C` 在函数开头就被 `repeat` **覆盖**了，所以 `CB` 与 `out_prev` 里用的
`C` 都是**展开后**的 `[b][l][h][n]`。

实测：`ngroups=2, nheads=4` 时，group 0 对应 head 0/1，group 1 对应 head 2/3。

### ★ 要点 4.2：`dt_segment_sum` 的方向

`dA_cumsum[..., :, None] - dA_cumsum[..., None, :]` ⇒
`decay[l][s] = exp(dA_cumsum[l] - dA_cumsum[s])`。

实测（chunk=4）的上三角为正、下三角为负：

```
[ 0.0,       0.064739,  0.178613,  0.333482  ]
[-0.064739,  0.0,       0.113874,  0.268743  ]
[-0.178613, -0.113874,  0.0,       0.154869  ]
[-0.333482, -0.268743, -0.154869,  0.0       ]
```

因 `dA < 0`，下三角 `exp <= 1`（衰减），上三角 `exp > 1` 但被 mask 清零 ⇒ **因果**。

### ★ 要点 4.3：`state_decay_out` 是 chunk 内逐位置值

`rearrange(dA_cumsum, "b h c l -> b c l h 1")` ⇒ `exp(dA_cumsum[l])`，
**不是** chunk 末的累积值。实测 chunk=2 时是 `[0.855344, 0.785010]`（两个不同值）。

### ★ 要点 4.4：`D` 是加性跳连，一维时 reshape 成 `[h][1]`

`D.dim() == 1` ⇒ `rearrange(D, "h -> h 1")` ⇒ 广播到 `headdim`。
`out = out + x * D`（**加**）。

### ★ 要点 4.5：`z` 是乘性 `silu` 门控

`return out if z is None else out * F.silu(z)`。
`F.silu(z) = z * sigmoid(z)`，**不是** `sigmoid(z)`。

## 5. 编排段 `ssd_chunk_scan_combined_ref` 的四个要点

官方代码（`ssd_combined.py:687-725`）：

```python
def ssd_chunk_scan_combined_ref(x, dt, A, B, C, chunk_size, D=None, z=None, dt_bias=None, dt_softplus=False):
    batch, seqlen, nheads, headdim = x.shape
    dstate = B.shape[-1]
    if seqlen % chunk_size != 0:
        dt = F.pad(dt, (0, 0, 0, chunk_size - seqlen % chunk_size))
    dt = rearrange(dt, "b (c l) h -> b h c l", l=chunk_size)
    dt = dt.float()
    if dt_bias is not None:
        dt = dt + rearrange(dt_bias, "h -> h 1 1")
    if dt_softplus:
        dt = F.softplus(dt)
    dA = dt * rearrange(A, "h -> h 1 1")
    dA_cumsum = torch.cumsum(dA, dim=-1)
    states = chunk_state_ref(B, x, dt, dA_cumsum)
    states_dtype = states.dtype
    if states.dtype not in [torch.float32, torch.float64]:
        states = states.to(torch.float32)
    states = rearrange(state_passing_ref(rearrange(states, "... p n -> ... (p n)"), dA_cumsum[:, :, :, -1])[0],
                       "... (p n) -> ... p n", n=dstate)
    states = states.to(states_dtype)
    out = chunk_scan_ref(B, C, x, dt, dA_cumsum, states, D=D, z=z)
    return out
```

### ★ 要点 5.1：`dt_bias` 在 `softplus` **之前**

官方顺序是 `dt = dt + dt_bias` 然后 `dt = F.softplus(dt)`。
所以 `dt_bias` 的最终影响是 `(dt + bias) * A` 之后再 cumsum。

★ 由于有 cumsum，`dA_cumsum` 的增量是 **`(l+1) * bias[hi] * A[hi]`**（累积效应），
不是常数 `bias * A`。

### ★ 要点 5.2：`dt_bias` 逐 head 一个标量

`rearrange(dt_bias, "h -> h 1 1")` ⇒ 广播到该 head 的所有 chunk 位置。

### ★ 要点 5.3：`(p, n)` 的展平与还原

`rearrange(states, "... p n -> ... (p n)")` 与
`rearrange(states, "... (p n) -> ... p n", n=dstate)` 是一对，
展平索引 = `p * n + ni`（`p` 高位、`n` 低位，C 序）。

### ★ 要点 5.4：`seqlen` 非整数倍 ⇒ 官方参考实现崩

`dt` 被 pad 到整数倍，但 `chunk_scan_ref` 的
`assert seqlen == nchunks * chunk_size` 用的是**原始 `seqlen`** ⇒ AssertionError。

实测：`seqlen=6, chunk=4` ⇒ `AssertionError`（无消息）。

## 6. 官方源码的三个复现坑

### 坑 1：`einops` 不在评测环境里

官方全篇用 `einops` 的 `rearrange` / `repeat`。`.venv-eval` 里**没有 einops**
（实测 `ModuleNotFoundError: No module named 'einops'`）。

⇒ 复现时必须把 `rearrange` / `repeat` 换成等价的 `reshape` / `permute` / `unsqueeze`，
**并且逐个核对索引布局**。本题踩过两个坑：

1. `repeat(B, "b l g d -> b l (g h) d", h=h//g)` 等价于
   `B.unsqueeze(3).expand(b,l,g,h,d).reshape(b,l,g*h,d)`（`g` 在前）；
2. `rearrange(decay, "b h c l s -> b c h l s")` 等价于 `permute(0,2,1,3,4)`。

### 坑 2：`F.pad` 的参数顺序

`F.pad(dt, (0, 0, 0, chunk_size - seqlen % chunk_size))` 里 `dt` 的形状是
`(b, l, h)` ⇒ pad 从**最后一维**往前数：`(0,0)` 作用在 `h` 维，
`(0, k)` 作用在 `l` 维 ⇒ **`l` 维补 `k` 个 0**。

### 坑 3：einsum 的求和索引要自己数

`einsum` 的规则是「出现在输入里但没出现在输出里的字母才是求和索引」。
`"bclhn,bhcl,bhcl,bclhp->bchpn"` 的求和索引只有 `l` ——
第一遍写的时候多求和了 `h`，在 `nheads == 1` 的数据上完全看不出来，
只有多 head 的用例才暴露。

## 7. 三层验证结构

1. **官方层**：从 upstream 三个 `*_ref` 文件用 `ast.get_source_segment` **逐字抽取**，
   只把 `einops` 的 `rearrange` / `repeat` 换成等价的 `reshape` / `permute`。
2. **参考解层**：`/tmp/mamba850_solution_staged.py`（纯 Python 嵌套列表 + `math.exp`）。
3. **oracle 层**：不依赖参考解、按官方算式独立重写一遍的纯 Python 实现。

实测 **486 个检查项**（18 组形状 × 4 个 mode × 若干 D/z/bias 组合）：
- `official-vs-ref <= 1.435e-07`（官方是 float32，参考解是 float64）；
- `oracle-vs-ref = 0.000e+00`（完全一致）。

另有 **9 条域约束断言**（`chunk_size <= 0`、`nheads % ngroups != 0`、
`seqlen` 非整数倍、`dt` 布局错、`D` 形状错、`seqlen != nchunks * chunk_size`、
官方 `seqlen` 非整数倍时抛异常等）。

**为什么用 float64 对 float32**：官方全程 float32（`dt.float()`），
而参考解用 Python `float`（float64）。差值在 `1e-7` 量级，
判题时用 `1e-9` 级别的容差即可区分「算错」与「精度差」。

## 8. mutation 的三条原则

本轮的 mutation 清单共 43 项，最终 **43/43 CAUGHT**。踩过的坑：

1. **等价变异要主动剔除**。首轮里有 2 个存活，逐一核对后确认是
   **真等价**：
   - `acc += sd * dv * xv` → `acc += dv * sd * x[...]`（乘法交换律）；
   - `row.append(out_intra + out_prev)` → `row.append(out_prev + out_intra)`
     （加法交换律）。

   判据是「**在全部合法输入域上输出相同**」，不是「这一组用例上相同」。

2. **校验层必须拆成互不依赖的独立防线**。例如 `chunk_scan` 那条 check 里同时写了：
   - 断言 `out == oracle`；
   - 断言 `scores_decay` 的上三角全为 `0`；
   - 断言 `decay[l][s]` 等于 `exp(dA_cumsum[l] - dA_cumsum[s])`；
   - 断言「z 改成加性」的候选**必须不同**；
   - 断言「D 改成乘法」的候选**必须不同**。

   只写第一条的话，一个把 `out + x*D` 写成 `out * D` 的实现可能「恰好」通过。

3. **不可观测的差异要设计专门的用例，且锚点必须落在真正会变的输入上**。
   本轮有两处典型：

   - **双向断言本身也要挑数据**：`group→head` 的 `(g h)` vs `(h g)` 顺序断言，
     在 `ngroups == 1` 的数据上两个候选**完全相等** ⇒ 断言误报。
     修法是把它挪到循环外，用一组**专门构造的多 group 数据**
     （`nheads=4, ngroups=2`）。
   - ★ **错误消息也是可观测差异**：首轮有两个变异存活 ——
     「删掉 `chunk_state` 的 seqlen 整除检查」与「删掉 `ssd_forward` 的整除检查」。
     根因是：删掉检查之后，**下游的形状校验 / `chunk_state` 仍会抛异常**，
     域约束用例只断言「抛了 `ValueError`」⇒ 区分不出来。
     修法有两步：
     ① 把域约束用例的输入改成**形状合法**（`dt` 给成 `(1,1,1,2)` 而不是 `(1,3,1)`），
     这样删掉检查后就不会被形状校验顶替报错；
     ② 断言**错误消息内容** —— `ssd_forward` 的报错消息必须带 `ssd_forward` 字样，
     以便与 `chunk_state` 代报的 `seqlen must be divisible by chunk_size` 区分开。

   教训：**「抛异常」不是一个足够细的判据，「抛的是哪个阶段的异常、消息说什么」才是。**

## 9. 与已有题的划界

- `la-mamba-selective-scan`（Mamba-1）：**逐 token 的 ZOH 离散递推**与门控跳连，
  没有 chunk 分解、没有 `dstate` 维状态矩阵、没有 `B`/`C` 分组矩阵。
- `la-gla-chunked`（GLA）：chunk 并行形式与朴素递推的**恒等账**，
  衰减是**逐 key 通道的标量**，状态是 `[K, V]`，没有 `dstate` 维、没有 `B`/`C`。
- `la-retnet-retention-xpos-ledger`（RetNet）：三种表示的**等价性**（不变量），
  本题是**具体三段计算口径**。
- `la-xlstm-stabilized-gate-ledger` / `la-slstm-normalizer-state-ledger`（xLSTM）：
  门控与 normalizer state，与 SSD 的 `dstate` 维状态矩阵无关。

★ 官方仓库里 `ssd_minimal.py` 在 `main` 分支已被删除（404），
三个 `*_ref` 现在分散在 `ssd_chunk_state.py` / `ssd_state_passing.py` / `ssd_chunk_scan.py`
三个文件里。

## 10. 官方源码位置索引

| 符号 | 位置 |
|---|---|
| `chunk_state_ref` | `ssd_chunk_state.py:1094-1122` |
| `chunk_state_ref` 的 `repeat(B, ...)` | `ssd_chunk_state.py:1114` |
| `chunk_state_ref` 的 `F.pad` 分支 | `ssd_chunk_state.py:1116-1118` |
| `chunk_state_ref` 的 `decay_states` | `ssd_chunk_state.py:1121` |
| `chunk_state_ref` 的 einsum | `ssd_chunk_state.py:1122` |
| `state_passing_ref` | `ssd_state_passing.py:327-350` |
| `state_passing_ref` 的 `cat` | `ssd_state_passing.py:339` |
| `state_passing_ref` 的 `F.pad` + `cumsum` | `ssd_state_passing.py:340-341` |
| `state_passing_ref` 的 `tril` mask | `ssd_state_passing.py:347-348` |
| `state_passing_ref` 的 einsum | `ssd_state_passing.py:349` |
| `state_passing_ref` 的返回切分 | `ssd_state_passing.py:350` |
| `chunk_scan_ref` | `ssd_chunk_scan.py:1846-1887` |
| `chunk_scan_ref` 的 `assert seqlen` | `ssd_chunk_scan.py:1864` |
| `chunk_scan_ref` 的 `CB` | `ssd_chunk_scan.py:1868-1870` |
| `chunk_scan_ref` 的 `dt_segment_sum` | `ssd_chunk_scan.py:1871-1872` |
| `chunk_scan_ref` 的 `tril` mask | `ssd_chunk_scan.py:1875-1876` |
| `chunk_scan_ref` 的交叉项 einsum | `ssd_chunk_scan.py:1877-1878` |
| `chunk_scan_ref` 的 `state_decay_out` | `ssd_chunk_scan.py:1879` |
| `chunk_scan_ref` 的 `out_prev` | `ssd_chunk_scan.py:1880-1881` |
| `chunk_scan_ref` 的 `D` 跳连 | `ssd_chunk_scan.py:1884-1886` |
| `chunk_scan_ref` 的 `z` 门控 | `ssd_chunk_scan.py:1887` |
| `ssd_chunk_scan_combined_ref` | `ssd_combined.py:687-725` |
| `ssd_chunk_scan_combined_ref` 的 `F.pad` | `ssd_combined.py:703-704` |
| `ssd_chunk_scan_combined_ref` 的 `dt_bias` | `ssd_combined.py:707-708` |
| `ssd_chunk_scan_combined_ref` 的 `softplus` | `ssd_combined.py:709-710` |
| `ssd_chunk_scan_combined_ref` 的 `dA` | `ssd_combined.py:711-712` |
