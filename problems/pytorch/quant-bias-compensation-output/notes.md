# 解题说明 · Bias Compensation 输出偏置的逐通道流式更新

## 出处

- Bias Compensation（Gong, Zheng, Hu, Lin, Fan, Zhang & Li, CAAI AIR 2024, arXiv:2404.01892, "Minimize Quantization Output Error with Bias Compensation"）：不去优化非凸的量化过程，而是在量化层**之后**补一个逐通道偏置，用校准数据上"浮点输出 − 量化输出"的差直接把它求出来；论文证明这个输出误差最小化是凸问题，因此不需要训练或微调。
- 本题的逐行口径转述自官方仓库 `GongCheng1919/bias-compensation` 的 `bias_compensation/quantizers/BiasCompensation.py`（2026-10-30 经 `codeload` tarball 取得，HTTP 200，与本文式子逐条对上）：`compute_shape` 里"负轴按 `output.dim()` 归一化 / 通道轴填尺寸、其余轴填 1 / 非通道轴按升序进 `mean_axis`"、`update` 里"先 `self.nsamples += 1` 再用 `(bias*(nsamples-1)+bias_update)/nsamples`"、`post_bias=True` 分支的 `float_output - quantized_output + self.bias.view(bias_shape)`、以及 `if len(self.mean_axis) > 0:` 才取均值这一句；`forward` 的 `output += bias.view(self.bias_shape)` 就是 `corrected_output` 的语义。
- **题源确认**：该条目在 HuangOwen/Awesome-LLM-Compression 的 `### Quantization` 章节内，带 `[[Paper]](arxiv 2404.01892) [[Code]](GongCheng1919/bias-compensation)`（2026-10-30 直接抓取 README 核对，该行在列表内）。

## 来源与刻意简化

- **多批校准的呈现方式改了**。上游是 `torch.nn.Module` + 前向钩子：每来一次校准前向就 `update()` 一次，状态（`bias`、`nsamples`）在两次调用之间隐式保持。本题把 $T$ 次前向一次性作为 `float_batches`/`quant_batches` 的第 0 轴给出，并额外返回 `bias_updates` 与 `bias_after` 两个轨迹，方便判题只看返回值；**每次调用计数只加 1** 这个上游行为被原样保留，所以校准 batch 越大并不会让该次更新更"有分量"。
- **`act_func` / `clone_output` 分支删除**。两者只是钩子包装（先加偏置再过激活、是否原地改输出），与本题要考的偏置估计无关；因此本题没有"加完偏置再算误差"的分支。
- **`channel_size` 从断言改成校验**。上游 `compute_shape` 用 `assert np.prod(bias_shape) == channel_size` 抛 `AssertionError`，本题按题库惯例改成 `ValueError`；`channel_size=None` 是本题新增（上游构造时必填），只为让"由形状自己推通道数"这一档也可测。
- **`init_bias` 与 `T=0` 是本题新增的边界**。上游的 `bias` 是构造时就 `zeros` 的 `nn.Parameter`，没有初值接口；也不接受空校准。本题允许 `init_bias`（$b_0$）与把 $T=0$ 判成非法，好让这两条可测。要注意它的效果分两种：`post_bias=False` 时第一次更新就把它的权重乘成 0，等于完全无效；`post_bias=True` 时它通过 $\Delta_0$ 里的当前偏置进入观测，且系数在后续每一步都保持 1，所以整条轨迹相对零初值恰好平移 `init_bias`。
- **`residual_mse` 是本题自设的度量**，源码里没有；上游只在实验里报下游任务指标。`corrected_output` 一律用**最终**偏置广播，而真实 `forward` 用的是"当时的"偏置（流式过程中每步都在变）——这是为了让返回值唯一而做的有意的非时序化。
- **内部 float64、出口按输入 dtype**。上游是 float32 的模块状态；本题把均值与流式平均放在 float64 里算，再 cast 回输入 dtype，使纯 Python oracle 与 torch 实现只在最后一步对齐。
- 不涉及：论文里的 `scale` 补偿旁路（源码中被注释掉）、`Ternary`/`fixedpointQ`/`ul2ptq` 等量化器、ViT/LLM 上的整层挂载与评测、任何需要真实模型前向的部分。

## 核心与易错点

- 负轴归一化的分母是**单次前向的秩** $d=\text{dim}-1$，不是整个张量的 `dim()`；$(T,B,C,S)$ 上 `channel_axis=-2` 指的是 $C$ 而不是 $B$。
- `bias_shape` 只描述单次前向（长度 $d$），第 0 轴 $T$ 不属于它；广播时若忘了这一层就会把 $T$ 当成通道数。
- `nsamples` **先自增后使用**：第一次是 $(b_0\cdot 0+u_0)/1=u_0$，旧值/初值被整个覆盖，第二次起才是真正的移动平均。写成 `(bias*(t-1)+u)/t`（0 基）会在 $t=0$ 时除零或算错权重。
- `post_bias=True` 里加回的是**上一次更新结束后的**偏置，不是本次的；把它当成 `bias_after[t]` 用就全错。
- `mean_axis` 为空时上游**不取均值**，直接展平——此时 `bias_updates[t]` 的长度是整张单次前向的元素数，而不是"每通道一个"。
- `bias_updates` 与 `bias_after` 是两个不同的量：前者是本次的观测，后者是观测与旧偏置加权之后的结果。判题分别检查，写成一个就露馅。
- `init_bias` 的作用完全由 `post_bias` 决定：`post_bias=False` 时 $\Delta_t$ 不含偏置，第一次更新就把初值的权重乘成 0，它对返回结果毫无影响；`post_bias=True` 时它进了 $\Delta_0$，且系数在每一步都保持 1，于是整条轨迹相对零初值恰好平移 `init_bias`。把它当成"普通的初始猜测"来做正则/收敛类假设是错的。
- 放大校准 batch 不改变结果（batch 轴被均值掉），但把 $T$ 从 2 增到 3 会改变每个后续样本的权重 $1/(t+1)$。

## 对拍口径

- 测试用独立的纯 Python `_oracle`：三层 `for` 手算差值、按 `mean_axis` 列表逐维求和除计数、再按 $1/(t+1)$ 加权，`math` 级别算术，与 torch 实现在 float64 下比对；float32 容差 `1e-5`、float64 容差 `1e-9`。
- 三个元组键（`channel_axis_norm` / `bias_shape` / `mean_axis`）用精确相等断言，专门覆盖负轴、多通道轴、`mean_axis` 为空三种形态。
- 另设专项：dtype 扫（float32/float64 结果一致到容差）、非连续视图（转置/切块后结果不变）、输入不被修改、非法输入（形状/dtype 不一致、空轴、越界轴、重复轴、`channel_size` 不符、`init_bias` 元素数不符、$T=0$）、退化探针（$F=Q$ 且初值为 0 时偏恒为 0、$T=1$ 时 `bias == bias_updates[0]`、`post_bias=False` 与 True 在 $T=1$ 时等价）。

## 复杂度

$O(T\cdot \lvert R\rvert)$（$\lvert R\rvert$ 是单次前向的元素数）：一次差值、一次按轴归约、一次长度 $C$ 的加权更新。无迭代求解、无矩阵乘、无显存峰值问题。

## 相关题目

- `quant-rtn-gptq-error`、`quant-adaround-recon-search`：都在改"量化带来的输出误差"，但前者做层内逐列误差再分配、后者搜索舍入方向；本题**不动量化器**，只在层后加一个偏置，是第三条路。
- `quant-hqq-scale-solve`：同样是闭式解，但求的是 scale；本题求的是输出偏置，且带流式计数这个坑。
- `quant-omniquant-transform-clipping`：OmniQuant 的等价变换也在削输出误差，但它是可学习参数的网格初值；本题的更新量是纯统计的、一步到位。
