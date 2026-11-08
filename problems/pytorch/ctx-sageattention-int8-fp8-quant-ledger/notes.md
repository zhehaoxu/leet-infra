# 官方机制与易错点笔记

## 官方来源与版本

- 仓库：`thu-ml/SageAttention`，默认分支 `main`（GitHub API trees 返回 159 blobs，truncated=False）。
- 用到的文件：`csrc/fused/fused.cu`（`QuantInt8Kernel` / `MeanScaleKernel` / `TransposePadPermuteKernel` / `quant_per_block_int8_cuda` / `mean_scale_fuse_quant_cuda` / `scale_fuse_quant_cuda` / `transpose_pad_permute_cuda`）、`csrc/numeric_conversion.cuh`（`float_to_int8_rn` / `floatx4_to_e4m3x4`）、`sageattention/triton/quant_per_block.py`（`quant_per_block_int8_kernel`）、`sageattention/quant.py`（`per_block_int8` / `per_channel_fp8`）。
- 回源方式：GitHub API `git/trees/main?recursive=1` + `contents?ref=main` + `base64.b64decode`。
- 官方对拍脚本用 `.venv-eval/bin/python`（torch 2.14.1）把两套 kernel **逐行展开成等价纯 Python**：CUDA 与 Triton 的 INT8 公式各一份 oracle，`MeanScaleKernel` 与 `TransposePadPermuteKernel` 各一份展开，共 **1800 组**（int8 600 / fp8 600 / permute 600）全部一致。

## per-block INT8 的两套口径

官方有**两份** per-block INT8 实现，数学上等价但逐点不等价：

| | CUDA `QuantInt8Kernel` | Triton `quant_per_block_int8_kernel` |
|---|---|---|
| `amax` | `max(0.0000001f, max\|x\|)` —— **有防零下界** | `max(\|x\|)` —— **无下界** |
| 舍入 | `cvt.rni.sat.s8.f32`（**round-to-nearest-even** + saturate） | `x_int8 += 0.5 * where(x_int8 >= 0, 1, -1)`（**round-half-away-from-zero**） |
| 缩放写法 | `x * (127/amax)`（`tmp_scale = 127.0f / s_amax`） | `x / (amax/127)`（`scale = tl.max(tl.abs(x)) / 127.`） |
| 越界 token | 越界位置 `x_val_float = 0.0f` 参与 amax 归约 | `tl.load` 用 `mask` 屏蔽，不参与 max 归约 |

- **RNE vs half-away-from-zero 的三个分水岭**：
  - 正数 `.5`：`2.5` 在 RNE 下给 `2`（取偶），在 away 下给 `3`。
  - 负数 `.5`：`-2.5` 在 RNE 下给 `-2`（取偶，`floor(-2.5) = -3` 是奇数故进位到 `-2`），在 away 下给 `-3`。
  - `-0.4`：away 口径先减 `0.5` 得 `-0.9`，**向零截断**得 `0`（不是 `-1`）。RNE 下 `floor(-0.4) = -1`、`diff = 0.6 > 0.5` 也给 `0`。
- **`sign` 判定用的是加 `0.5` 之前**的浮点值，这是 Triton 源码里 `tl.where(x_int8 >= 0, 1, -1)` 的语义。
- **`sm_scale` 先乘在值上**（`x_val_float[i][j] *= sm_scale`），它同时进入 `amax` 归约和量化，不是乘在量化后的整数上。
- **`sm_scale` 的 `1.44269504`**：官方 `per_block_int8` 对 q 侧传 `sm_scale * 1.44269504`（`log2(e)`）、对 k 侧传 `1.0`。因为 Triton 注意力 kernel 用 exp2 实现 softmax。这个系数由调用方决定，本题只把 `sm_scale` 当普通缩放系数。
- **分块边界**：`q_scale` 形状 `(qo_len + BLKQ - 1) // BLKQ`（`BLKQ = 128`）、`k_scale` 形状 `(kv_len + BLKK - 1) // BLKK`（`BLKK = 64`）；最后一块不满时只有实际 token 参与量化。
- **饱和在本题数值域内不可达**：`|x| <= amax` 且 `scale = amax/127`，所以 `|x_int8| <= 127`，`cvt.rni.sat.s8.f32` 的 saturate 分支永不触发。写 mutation 时不要在这上面浪费力气。

## per-channel FP8（E4M3）的坑

- **`MeanScaleKernel` 是 per-channel 的**。`dim3 grid(num_heads, batch_size, head_dim)`，`d_id = blockIdx.z`，每个 `head_dim` 通道**独立**沿 token 维归约。这是与「per-tensor 量化」最刺眼的差别，也是本题最主要的判题点。
- **`fp8_padded_num_tokens = (num_tokens + 15) / 16 * 16`**：均值分母是 **padding 到 16 的长度**，不是真实 token 数，也不是 padding 到 64 的长度（那是输出 tensor 的形状）。
- **padding 位置的值是 `0`**，由上游 `TransposePadPermuteKernel` 的 `cp_async` `kFillZero` 填入，且**参与 `max` / `min` / `sum` 归约**。因此全正通道的 `min_val` 会被拉到 `0`，进而改变 `amax`。
- **`smooth_v=True` 时 `amax` 是减均值之后的**：`s_amax_val = fmaxf(fabsf(block_max_val - s_mean_val), fabsf(block_min_val - s_mean_val))`。写成 `max(|max|, |min|)` 就错了。
- **`smooth_v=False` 时不返回 mean**（官方 `scale_fuse_quant_cuda` 传的 `mean` 是 `nullptr`，返回的第三元组是 `None`），且 `amax` 用 `max(|max|, |min|)`。
- **`scale` 与 `recp_scale` 互逆**：`scale = s_amax_val / scale_max`（`scale_max = 448.0`，E4M3 上界）、`recp_scale = scale_max / s_amax_val`。量化时乘的是 `recp_scale`（不是除 `scale`）。
- **FP8 字节本身不入判题域**：`floatx4_to_e4m3x4` 用 `cvt.rn.satfinite.e4m3x2.f32`（硬件舍入 + 饱和到有限值），CPU 上不可精确复现。本题只判 `mean` / `amax` / `scale` / `recp_scale`。
- 官方 `per_channel_fp8` 的返回值顺序是 `(v_fp8, v_scale, vm)`，`vm` 在 `smooth_v=False` 时是 `None`。

## 转置-填充-置换的坑

- **`padded_num_tokens = (num_tokens + pad_size - 1) / pad_size * pad_size`**，`pad_size` 即 `CTA_SIZE = 64`。
- **输出布局**：`tensor_layout="HND"` 时是 `(b, h_kv, head_dim, padded_num_tokens)`，即 token 维被换到最后。
- **置换算式**：`smem_load_row = base + (mod / 8) * 2 + ((mod / 2) % 4) * 4 + (mod % 2)`，其中 `base = ((thread_id / num_threads_per_token) / 16) * 16` 是 **16 的倍数**，`mod = (thread_id / num_threads_per_token) % 16`。所以置换是 **16 个一组的组内置换**，不是 64 个一组。
- **算式逐项展开 `mod = 0..15` 得 `0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15`**，与 kernel 注释 `// 0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15` **完全一致**。而官方 `per_channel_fp8` 的 docstring 写的是 `0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15`，那是**逆置换**（`out[j] = src[L1[j]]`）——两者描述的是同一布局，只是方向相反。本题以 kernel 算式为准（源 → 目标）。
- **padding 位置的值是 0**（`cp_async::SharedMemFillMode::kFillZero`），参与后续 FP8 统计。

## 验证方式与判题设计

- 官方回源对拍：**int8 600 组 + fp8 600 组 + permute 600 组 = 1800 组全部一致**。int8 部分 CUDA 与 Triton 口径交替覆盖，网格专门造 `.5` 边界（`-1.0, -0.5, 0.0, 0.5, 1.0, 2.5, -2.5`）以暴露两种舍入的差异。
- 判题用例：3 个 `check_*`，各含独立纯 Python oracle、手算锚点与网格对拍。
- 手算锚点钉住：RNE 的 `.5` 取偶（`2.5 -> 2`）、away 的 `.5` 进位（`2.5 -> 3`）、负数 `.5`（`-2.5` 分别给 `-2` 与 `-3`）、`-0.4` 向零截断为 `0`、`amax_floor` 在全零输入下给出 `1e-7` 与 `0` 两种、末块不满时各块独立算 `amax`、`sm_scale` 同时影响 `amax` 与量化、per-channel 的两通道独立归约、均值分母是 padding 到 16 的长度、padding 的 `0` 参与 `min` 归约、`smooth_v=False` 时 `mean` 为 `None`、16 个 token 全正且无 padding 时 `min` 是真实最小值、置换索引序列、16 元组置换方向、padding 到 64 与 16 的区别、转置后行=head_dim。
- mutation testing：49 个单点变异，全部 CAUGHT，且**正确解双模式自检先行确认 ACCEPTED**。

## 对拍与 mutation 过程中踩到的坑（可复用经验）

1. **对拍脚本的比较器要支持 `inf`**。`recp_scale` 在 `amax == 0` 时是 `inf`，`abs(inf - inf) = nan` 会让断言误判。必须先判 `math.isinf`。
2. **对拍脚本的期望值要跟解的字段集完全一致**。第一版解暴露 `saturated` 字段，但 `|x| <= amax` 恒成立 ⇒ 量化值永不越界 ⇒ `saturated` **恒为 False**，是个死字段。改成暴露 `scaled`（进舍入器之前的浮点值）才有区分度——它把「乘 `127/amax`」与「除 `amax/127`」两种写法暴露出来。
3. **官方源码复核会推翻自己的设计**。第一版 `fp8_channel_scale` 做成了**全局**归约，复核 `MeanScaleKernel` 的 `dim3 grid(num_heads, batch_size, head_dim)` 后才发现它是 **per-channel**，整个 mode 的语义都得改写。**先读完 kernel 再写解**，不要凭 API 名字猜。
4. **手写 oracle 也要独立于解的写法**。oracle 里 `mean = None` 与 `means.append(None)` 是两处代码，只改一处会造成「变异无效」的空转。
5. **真等价变异必须剔除并换成可测形态**。首轮 48 变异有 7 个存活，诊断后确认：
   - `rne-sat-lower`（饱和下界）：`|x_int8| <= 127` 恒成立 ⇒ 下界饱和分支不可达；
   - `away-sign-flip`（`v >= 0` 改 `v > 0`）：只在 `v == 0.0` 时不同，而 `0.0` 两边都截断为 `0`；
   - `int8-cuda-mul-form` / `int8-triton-div-form`：`f * (127/amax)` 与 `f / (amax/127)` 浮点近等价，窗口太窄；
   - `fp8-nosmooth-amax`：`abs(v - 0.0) == abs(v)`；
   - `fp8-nosmooth-mean-not-none`：锚点改的是死变量 `mean`，而输出用的是 `means.append(None)`；
   - `fp8-min-init`：`max_val = vals[0]` 会被随后的循环覆盖成同样的值。
   逐个换成可测形态后 49/49 全 CAUGHT。
6. **网格的多个维度要正交**。fp8 网格最初用 `style = case % 4` 配 `smooth = case % 2`，两者恒同相（`style` 为 2/3 时必然 `smooth=True`），导致「非 smooth 分支」的一半变异测不出。改成 `style = case % 5` 配 `smooth = (case // 5) % 2` 之后才覆盖全。
7. **初值类变异需要专门造样例**。`max_val = 0.0` / `min_val = 0.0` 这类变异只有在「全负通道」或「16 个 token 全正且无 padding」时才可测，补了两个手算锚点才抓住。

## 不进判题域的边界

attention 前向与 online softmax、累加顺序、CUDA kernel 的 tile / 线程映射与 `cp_async` / shared memory / `ldmatrix` 优化、FP8 字节的 `cvt.rn.satfinite.e4m3x2.f32` 硬件舍入、Triton 编译与 launch 配置、`tensor_layout` 的 NHD 分支、`per_warp_int8` 的 warp 变体、`SubMeanKernel` 与 `km`（key mean）路径、真实权重矩阵，全部不进判题域。本题只保留三段确定性的纯数值账。
