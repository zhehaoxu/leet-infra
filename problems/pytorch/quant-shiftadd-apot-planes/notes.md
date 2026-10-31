# 解题说明 · ShiftAddLLM 位平面标度的 ApOT 移位分解

## 出处

- **ShiftAddLLM: Accelerating Pretrained LLMs via Post-Training Multiplication-Less Reparameterization**（You, Guo, Fu, Zhou, Shi, Zhang, Kundu, Yazdanbakhsh, Lin；Georgia Tech / Intel Labs / Google / Google DeepMind；NeurIPS 2024，**arXiv:2406.05981**，仓库 `GATECH-EIC/ShiftAddLLM`）：把预训练权重的乘法换成"按二进制平面选符号 + 移位 + 加法"，ApOT（additive powers of two）负责让每平面的标量本身也只由少数几个 2 的幂组成。
- 本题的公式逐行转写自该仓库 `quantizers/bcq_quant/bcq_shift.py`（2026-10-30 由本地 tarball 复核行号）：
  - `get_shift_and_sign` **L15-24**：`sign = torch.sign(x)`、`shift = round(log(|x|)/np.log(2))`（`rounding` 默认 `deterministic`，即 `x.round()`，**半偶舍入**）。
  - `round_power_of_2` **L26-39**：`q_bias`/`scale` 都是 `None` 时只剩 `(2.0 ** shift) * sign`。
  - `additive_power_of_2` **L41-48**：首项 `x_rounded = 2**shift * sign`，随后 `nums-1` 次都对 `x - x_rounded`（**已提交和的残差**）再取一项并累加。
  - `greedy_mean_torch` **L131-161**：`b = r.sign()`、`alpha = (r.abs()*wf).sum(dim=1)/torch.sum(wf, dim=1)`、`r -= b*alpha`；`wf is not None` 分支里 `shift=False` 时不做任何幂次处理。
  - `quantize_shift` **L106-107 / L109-110 / L115**：`group_size = group_size if group_size > 0 else orig_shape[-1]`、`w_ = w_.view([-1, group_size])`、`wf = torch.ones(w_.shape)`、最后 `ret = torch.einsum('ijl,il->ij', (B, alpha))`。
- **仓库状态**：`README.md` 开头声明该仓库已不再维护，最新工作是 `ShiftAddQwen`（换拟合流程与 LUT-free kernel，保留二进制平面格式），但本仓库仍是 NeurIPS 2024 论文的参考实现——本题需要的四个函数都在 `bcq_shift.py` 里原样存在。

## 来源与刻意简化

- **上游的 ApOT 只出现在精化环节，本题把它接到贪心标度上——这是本题的组合作图**。原文 `additive_power_of_2` 的唯一调用点是 `refine_mean_torch` **L177**，喂给它的是 `batch_cg_torch` 解出、再 `abs().sort(descending=True)`（L175）的 `Alpha_new`。CG 求解器和排序都不在本题范围内（见下条），于是本题对 `greedy_mean_torch` 的 `alpha` 直接做 ApOT；**不做** `abs()` 也不按降序排（贪心标度本就非负且首项最大，排序是恒等操作，但绝对值会改变残差符号的语义，不能顺手加）。判题契约以题面为唯一规范。
- **`rounds=15` 的精化循环、`batch_cg_torch`（L226）与 `find_B_torch`（L200）整体排除**。前者是共轭梯度 + BST 枚举，依赖迭代收敛与 GPU 缓存管理，数值上随环境漂移，不适合做逐位判题；`use_bst=False` 的那条分支虽然只是重新贪心解一次平面，但它吃的是精化后的 `Alpha_new`，与本题的 `planes` 来源不同。
- **`q_bias` 与 `scale` 分支排除**。`round_power_of_2(x, q_bias=..., scale=...)` 会先减/除再幂次、最后加/乘回来，而 `quantize_shift` 从不传 `shift=True`，`get_best_bias`(L51)/`get_best_scale`(L58) 在仓库里也没有活的调用点（只有注释掉的旧签名）。本题不臆造"缩放域上分解再乘回"的复合口径。
- **`rounding="floor"` 与 `"stochastic"` 分支排除**。`get_shift_and_sign` 有 `torch.floor` 这条支路、`round()` 有 `torch.bernoulli` 那条随机支路，但仓库里没有任何调用方传过非默认值，且随机支路破坏确定性。本题只保留默认的确定性半偶舍入。
- **`transpose` 参数排除**。上游 `quantize_shift` 用 `transpose=True` 时先 `.transpose(1,0).contiguous()` 再切组；本题输入就是 2-D 矩阵，`view`/`reshape` 一律按行主序，没有列主序口径。
- **`alpha_terms`（`apot_terms`）是本题新增的快照**。上游 `additive_power_of_2` 只 `return x_rounded`，中间项不落盘；本题把它们返回出来，判题才能定位是"第一项取错幂"还是"残差链写错"。
- **内部 float64，输出按输入 dtype**。上游全程 float32 CUDA；本题把 `log`/`round`/`2**shift` 都放在 float64 里做，最后一次转回输入 dtype，这样纯 Python 标量 oracle 与 torch 实现只在最终转 dtype 处对齐。
- 范围之外：LLM 层挂载与 `zeroShot`/`lut_gemm`/`energy` 评测、`bcq.py` 的 `BCQ` 权重包装、`quantizer.py` 的 `assert alpha 降序`、以及任何需要真实模型前向的部分。

## 核心与易错点

**分组吃的是展平后的序列，不是行。** `g` 不整除 `C` 时组会跨行：$(2,3)$ 配 `group_size=2` 的三组是 $[1,2]$、$[4,8]$、$[16,32]$，于是同一行的第 2、3 列用不同标度、第 3 列和第 4 列（下一行首）共用一个标度。把组形状写成 $(R, C/g, g)$ 是错的，正确是 $(R\cdot C/g, g)$。

**取整发生在 $\log_2$ 之后，而且是"就近取整"不是 floor。** `torch.round` 把 $\log_2$ 取到最近的整数（恰好在两个整数正中间时取偶数）：$\alpha=3$ 时 $\log_2 3=1.585\to 2\to 4$，写成 floor 会得到 $2$；$\alpha=0.4$ 时 $\log_2 0.4=-1.322\to -1\to 0.5$，floor 给 $0.25$。这两处是 round/floor 的分水岭，也是本题最容易丢分的一步。真正的并列点只可能出现在 $\alpha=2^{k+1/2}$（无理数）上，二进制输入踩不到，所以"取偶"这条规则本身基本不可观测——但别因此改用 floor。

**零项永远停在 0，而且会传染。** `sign(0)=0` 且 `log(0)=-inf`、$2^{-\infty}=0$，所以 $\alpha=0$ 或残差恰为 0 时那一项是 $0$，不是某个 ±1 的幂。位平面上同理：某个元素一旦落到 0，之后所有平面的 $b$ 都是 0，重建值被前面的平面完全决定（例一第 1 行第 3 个元素就是这条）。

**ApOT 的第 $k$ 项吃残差，不是原标度。** $d_k=\alpha-\sum_{m<k}t_m$，所以 $t_1$ 可正可负、可绝对值小于 $t_0$：$\alpha=1.5\Rightarrow[2,\,-0.5]$。若每步都重新对 $\alpha$ 取最近幂，两项都会变成 2，`alpha_apot` 就成了 4。

**重建用的是 `alpha_apot`，不是 `alpha`。** 这是整道题的意义所在（标度要能移位得到），也是最容易的失分点：`planes` 完全由 `alpha` 决定、`recon`/`mse` 完全由 `alpha_apot` 决定，两者搞混时 `planes`/`alpha` 还是对的，只有 `recon`/`mse` 错，容易被"前几个键对了"骗过去。

**`wf` 全 1 时那个 `alpha[torch.isnan(alpha)] = 0.` 永不触发。** 分母是 `torch.sum(wf, dim=1) = g > 0`，分子是若干非负数之和，不可能 NaN。它只在稀疏 `wf`（剪枝掩码）下才有意义，而稀疏掩码不在本题范围内——所以看到有人专门写 NaN 兜底，要意识到那是照抄了一个不可达分支。

**$S$ 只影响标度，不影响平面。** `num_bits` 决定 `planes` 的形状与内容；增大 `apot_nums` 只会把 `alpha_apot` 拉回 `alpha`，`planes`/`alpha` 逐位不变。反过来，`group_size` 改变分组会同时改变 `planes`、`alpha` 与 `recon`。
