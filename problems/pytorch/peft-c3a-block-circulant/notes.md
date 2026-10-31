# 解题说明 · C3A 的块循环卷积

## 出处

- `huggingface/peft` 的 `src/peft/tuners/c3a/utils.py`：`get_circulant_fast(w)` 把 `(m, n, bs)` 的核张量变成 `(m*bs, n*bs)` 的稠密增量阵，`BlockCircularConvolution` 是它的自定义 autograd 版本，用两次 FFT 把循环矩阵乘法换成逐元素乘积。
- `src/peft/tuners/c3a/layer.py`：`C3ALayer.get_delta_weight` 返回 `get_circulant_fast(kernel) / base_layer.weight.size(-1)`（即除以 `in_features`），`update_layer` 强制核张量形状为 `(out_features // block_size, in_features // block_size, block_size)`；`C3ALinear.forward` 的未合并路径是 `base(x) + BlockCircularConvolution(x, kernel) / x.size(-1)`。本课逐行对齐这两处。
- 论文：Chen 等，*Parameter-Efficient Fine-Tuning via Circular Convolution*，arXiv:2407.19342（C3A）。公式与本课口径均以源码为准。

## 核心与易错点

**块内下标是 `(j - i) % bs`，方向不能反。** 循环矩阵的定义是每行相对上一行左移一位，所以 `C[i, j] = k[(j - i) % bs]`：第 0 行正好是核本身。写成 `(i - j) % bs` 得到的是右移版本，等价于把核做了转置——`bs=2` 时两版只差非对角元，`bs>2` 时整块错位，是本题最容易过不了 `check_orientation` 的地方。

**归一化除的是 `d_in`，不是 `bs`。** peft 源码里分母来自 `weight.size(-1)`，也就是卷积输入的最后一维；`check_scaling_by_in_features` 用同一组核值把 `d_in` 从 4 扩到 8，增量恰好减半，如果按 `bs` 归一化就完全看不出这个差别。概念上它扮演 LoRA 的 `lora_alpha / r`。

**核张量的两个轴分别跟着输出块和输入块。** 形状是 `(out//bs, in//bs, bs)`，第 0 维是行块号 `m`、第 1 维是列块号 `n`。把二者写反在方阵上只是转置错误，但前向 `x @ deltaᵀ` 与 `check_block_locality`（只改 `kernel[1,0]` 只让 `delta` 的 `(1,0)` 块变化）会立刻暴露。

**`bs = 1` 时循环退化成普通稠密阵。** 每块只剩一个元素、`(j - i) % 1` 恒为 0，于是 `delta = kernel[:, :, 0] / d_in` 是一张完整的 `out × in` 矩阵——参数量优势消失。反过来 `bs = d_in = d_out` 时只有一块，整张增量就是一张循环阵，第 `o` 行等于核向右移 `o` 位。这两个极端各有一个检查卡住。

**常数核给出常数阵。** 核的 `bs` 个值全相等时，块内循环移位不改变任何元素，整张 `delta` 每行每列都是 `c / d_in`；此时每行元素和为 `c`。这个性质常用来快速验证归一化分母。

**为什么循环矩阵能省算力。** 循环阵可被 DFT 对角化：`C = Fᴴ diag(fft(k)) F`，所以一次块内乘法在频域里是逐元素乘、复杂度从 `O(bs²)` 降到 `O(bs log bs)`，这也是 peft 用 `fft`/`ifft` + `einsum` 而不是显式拼块的原因。判题口径要的是显式 `delta`，所以本题仍然物化——但 notes 里值得记住：C3A 的存储省在参数量 `out × in / bs`，算力省在频域逐元素乘，两者是不同的事。

**peft 的核固定 float32。** 源码注释说明 FFT 对 fp16/bf16 支持不完整，核张量被强制成 fp32，前向也在 fp32 路径上跑；本课题面要求内部 float64、返回 float32，与真实 peft 对拍时只能对到 `1e-7` 量级而不是逐位相等。

## 对拍口径

- `test_cases.py` 里的 `_oracle` 用 numpy 四层 `for m, n, i, j` 逐元素重算 `delta`，循环下标用手写的 `while src < 0: src += bs` 取模，与题面的向量化 `permute/reshape` 实现不共享任何代码；`check_against_oracle` 覆盖 `1×1×1`、非方阵、`bs=1`、`bs=in/2`、带偏置与纯 float32 输入等 6 组配置。
- 结构类检查各盯一个易错点：`check_block_layout` 验证块内只有 `bs` 个不同取值且其余块全零；`check_orientation` 用 `bs=3` 区分 `(j-i)` 与 `(i-j)`；`check_single_block_circulant` 验证整块循环移位方向；`check_scaling_by_in_features` 验证分母；`check_block_locality` 验证轴的语义；`check_merge_equivalence` 验证 `y = x @ (W0 + delta)ᵀ + b0` 且 `delta` 与 `x` 无关。
- 编写时另用真实 peft 复核：构造 `C3ALinear(nn.Linear(in, out), ...)` 后把 `c3a_kernel` 直接覆盖成随机核，与 `solution.solve` 对 6 组 `(in, out, bs)` 比较，`delta` 最大偏差 `7.5e-9`、前向最大偏差 `4.8e-7`（偏差全部来自 peft 的 fp32 FFT 路径）。该复核脚本不属于判题文件。

## 复杂度

设 `N = prod(x.shape[:-1])`。物化 `delta` 需要 `O(d_out × d_in)` 时间与显存（这也是返回值的固有代价，真实 C3A 只存 `d_out × d_in / bs` 个参数）；前向是两次稠密乘加，`O(N × d_out × d_in)`。若改用频域路径，块内乘法降到每块 `O(bs log bs)`，总训练侧参数与计算都按 `1/bs` 缩放。数值上全程 float64，只在返回时降回 float32。
