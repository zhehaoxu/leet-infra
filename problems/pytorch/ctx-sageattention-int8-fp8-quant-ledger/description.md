# SageAttention 的 per-block INT8 与 per-channel FP8 量化、舍入口径与转置置换账

## 背景

SageAttention 把注意力算子里的两个大矩阵（query / key）先量化成 INT8 或 FP8，再喂给 Tensor Core。量化这一步的数值细节决定了最终精度，而这些细节在 CUDA kernel 与 Triton kernel 里是**两套写法**：

- **CUDA kernel**（`csrc/fused/fused.cu`）用硬件指令 `cvt.rni.sat.s8.f32` 做舍入，是 round-to-nearest-**even**；
- **Triton kernel**（`sageattention/triton/quant_per_block.py`）用 `x += 0.5 * sign(x)` 再截断，是 round-half-**away-from-zero**。

两者在 `.5` 边界上给出不同的整数，在负数上差得更远。FP8 那条路则是 **per-channel** 的：沿 token 维对每个 `head_dim` 通道独立统计 `mean` / `amax`，再推出 `scale` 与 `recp_scale`。而 FP8 的 mma 布局还要求先做一次「转置 + 补齐 + 16 元组 tile 置换」。

本题重建这三段**纯数值**账：per-block INT8 的两套舍入口径、per-channel FP8 的统计推导、以及转置-填充-置换的布局。attention 前向、CUDA/Triton kernel 调度、FP8 字节的硬件舍入都不在判题域内——输入一律是已经算好的数值列表。

## 通用约定

- 所有输入都是 Python `float` 组成的列表（`x` 是二维列表：token × head_dim）；不涉及张量、梯度和批处理。
- 浮点比较采用绝对误差 `1e-9` 加相对误差 `1e-9`。
- `head_dim` 记作 `H`，即通道数；`tokens` 记作 `L`，即 token 数。
- 返回值统一是 `dict`，字段名见各 mode 说明。
- 下标一律从 0 开始，且**闭区间**。
- 所有 mode 都通过单入口 `solve(mode, **kw)` 调用。

## 模式与规则

### 1. `int8_block_quant`：per-block INT8 量化

入参：

| 参数 | 含义 | 默认值 |
|---|---|---|
| `x` | 二维列表，长度 `L`，每行长度 `H` | 必填 |
| `blk` | 每块的 token 数 | `128` |
| `sm_scale` | 缩放系数（先乘在值上） | `1.0` |
| `rounding` | `"rne"`（CUDA）或 `"away"`（Triton） | `"rne"` |
| `amax_floor` | `amax` 的防零下界 | `0.0000001` |

计算步骤：

```
# 1. 先把 sm_scale 乘在每个值上
f[t][d] = x[t][d] * sm_scale

# 2. 按 token 维分块，每块独立算 amax 与 scale
num_blocks = (L + blk - 1) // blk
for b in range(num_blocks):
    lo = b * blk
    hi = min(lo + blk, L)                       # 最后一块不满
    amax[b] = max(amax_floor, max(|f[t][d]| for t in [lo,hi), d in [0,H)))
    scale[b] = amax[b] / 127

# 3. 逐点量化
rounding == "rne":                              # CUDA 口径
    s = f * (127 / amax)
    q = round_to_nearest_even(s)                # 再 saturate 到 [-128, 127]
rounding == "away":                             # Triton 口径
    s = f / scale                               # 等价于 f * (127 / amax)
    q = trunc(s + 0.5 * sign(s))                # sign 用加 0.5 之前的 s
```

返回字段：

- `amax`：长度 `num_blocks`，每块的 `amax`；
- `scale`：长度 `num_blocks`，每块的 `scale`；
- `scaled`：长度 `L * H`，**进舍入器之前的浮点值**（按 token 优先展平）；
- `x_int8`：长度 `L * H`，量化后的整数（按 token 优先展平）；
- `num_blocks`：块数；
- `head_dim`：通道数。

易错点：

- **两种舍入口径在 `.5` 边界上不同**。`round_to_nearest_even` 在 `x.5` 处取**偶数**（`2.5 -> 2`，`3.5 -> 4`）；`trunc(s + 0.5*sign(s))` 是 half-away-from-zero（`2.5 -> 3`，`-2.5 -> -3`）。
- **`sign(s)` 用的是加 `0.5` 之前**的 `s`。`-0.4` 走负号分支得 `-0.9`，向零截断得 `0`（不是 `-1`）。
- **Triton 的向零截断**：`trunc(-0.9) = 0`，`trunc(-1.0) = -1`。Python 的 `int()` 恰好是向零截断，可直接用。
- **CUDA 有 `amax` 的 `1e-7` 下界**，Triton 没有。全零输入时 CUDA 的 `amax` 是 `1e-7`、Triton 是 `0`（此时 `scale` 为 0，量化值按 `0` 记）。
- **`sm_scale` 先乘在值上**，它同时影响 `amax` 与量化结果——不是乘在量化后的整数上。
- 官方 `per_block_int8` 对 q 侧传 `sm_scale * 1.44269504`（`log2(e)`）、对 k 侧传 `1.0`；这个系数由调用方决定，本题只把 `sm_scale` 当普通缩放系数。
- **`scaled` 是暴露中间量的关键字段**：它把「乘 `127/amax`」与「除 `amax/127`」两种写法的差异暴露出来，虽然两者数学等价。

### 2. `fp8_channel_scale`：per-channel FP8（E4M3）量化统计

入参：

| 参数 | 含义 | 默认值 |
|---|---|---|
| `x` | 二维列表，长度 `L`，每行长度 `H` | 必填 |
| `scale_max` | FP8 的最大幅值（E4M3 上界） | `448.0` |
| `smooth_v` | 是否减均值平滑 | `True` |

计算步骤：

```
# 1. 归约区间长度：padding 到 16
padded_tokens = ((L + 15) // 16) * 16
pad_count = padded_tokens - L

# 2. 每个 head_dim 通道独立归约（沿 token 维）
for d in range(H):
    vals = [x[t][d] for t in range(L)] + [0.0] * pad_count
    max_val = max(vals)
    min_val = min(vals)
    sum_val = sum(vals)
    if smooth_v:
        mean[d] = sum_val / padded_tokens
        amax[d] = max(|max_val - mean[d]|, |min_val - mean[d]|)
    else:
        mean[d] = None
        amax[d] = max(|max_val|, |min_val|)
    scale[d] = amax[d] / scale_max
    recp_scale[d] = scale_max / amax[d]
```

返回字段：

- `mean`：长度 `H`；`smooth_v=False` 时每个元素都是 `None`；
- `amax`：长度 `H`；
- `scale`：长度 `H`；
- `recp_scale`：长度 `H`；`amax == 0` 时为 `inf`；
- `padded_tokens`：padding 后的长度；
- `head_dim`：通道数。

易错点：

- **是 per-channel，不是 per-tensor**：每个 `head_dim` 通道沿 token 维**独立**归约。`x = [[1.0, 10.0], [3.0, 20.0]]` 的两个通道得到完全不同的 `mean` 与 `amax`。
- **均值分母是 `padded_tokens`（padding 到 16 的长度），不是真实 token 数**。`L = 1` 时分母是 `16`，`mean` 是 `sum / 16`。
- **padding 位置的值是 `0`，且参与 `max` / `min` / `sum` 归约**。所以全正的通道 `min_val` 会被拉到 `0`（`L` 不是 16 的倍数时）。
- **`smooth_v=True` 时 `amax` 是减均值之后的**：`max(|max_val - mean|, |min_val - mean|)`，不是 `max(|x|)`。这是最容易写错的一步。
- **`smooth_v=False` 时 `mean` 是 `None`**（官方不返回均值张量），且 `amax` 用 `max(|max_val|, |min_val|)`。
- **`scale` 与 `recp_scale` 互逆**：`scale = amax / scale_max`，`recp_scale = scale_max / amax`。量化时乘的是 `recp_scale`。
- `padded_tokens` 是 padding 到 **16**，不是 padding 到 64（64 是输出维度的补齐，见 mode 3）。

### 3. `transpose_pad_permute`：转置 + 补齐 + FP8 mma 布局置换

入参：

| 参数 | 含义 | 默认值 |
|---|---|---|
| `x` | 二维列表，长度 `L`，每行长度 `H` | 必填 |
| `pad_size` | token 维补齐的粒度 | `64` |

计算步骤：

```
# 1. token 维补齐到 pad_size 的倍数，padding 位置填 0
padded_tokens = ((L + pad_size - 1) // pad_size) * pad_size

# 2. 转置成 head_dim × padded_tokens
padded[d][t] = x[t][d]          # t < L 时取原值，其余为 0

# 3. 按 16 个一组做组内置换
#    源（组内位置 mod）-> 目标（组内位置 dest）：
#    dest = base + (mod // 8) * 2 + ((mod // 2) % 4) * 4 + (mod % 2)
#    base 是 16 的倍数
permuted[d][base + dest_off] = padded[d][base + mod]
```

返回字段：

- `padded_tokens`：补齐后的长度；
- `padded`：二维列表，`H` 行 × `padded_tokens` 列，是**补齐但未置换**的转置结果；
- `permuted`：二维列表，`H` 行 × `padded_tokens` 列，是**置换后**的结果；
- `permute_index`：长度 16 的整数列表，即组内置换的偏移表。

`permute_index` 逐项展开（`mod = 0..15`）为：

```
0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15
```

易错点：

- **置换是 16 个一组的组内置换**，不是 64 个一组，也不是全局置换。
- **置换方向是「源 → 目标」**：`permuted[d][base + permute_index[mod]] = padded[d][base + mod]`。反过来写（把 `permute_index` 当成「目标 → 源」）得到的布局是逆置换，虽然也是合法排列，但**不是**官方那个。
- **补齐粒度 `pad_size` 与 FP8 归约的 padding 到 16 是两回事**：前者是输出 tensor 的形状（默认 64），后者是统计区间长度（固定 16）。
- **padding 位置填 0**，参与后续 FP8 统计。

## 判题范围

- 只判上面三段纯数值账：`int8_block_quant` / `fp8_channel_scale` / `transpose_pad_permute`。
- **不判**：attention 前向、online softmax、累加、CUDA kernel 的 tile 与线程映射、`cp_async` 与 shared memory、`ldmatrix` 优化、FP8 字节的 `cvt.rn.satfinite.e4m3x2.f32` 硬件舍入、Triton 编译与 launch 配置、`tensor_layout` 的 NHD 分支、`per_warp_int8` 的 warp 变体、真实权重矩阵。
- 题面里的 `x` 只是一份数值列表，**不含任何真实权重**。
