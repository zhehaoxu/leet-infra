# Notes: VQ-Diffusion 反扩散单步的 log 空间转移矩阵

## 出处

- diffusers v0.42.1 `src/diffusers/schedulers/scheduling_vq_diffusion.py`（全文 516 行）。类 `VQDiffusionScheduler` 在 142-516，`VQDiffusionSchedulerOutput` 在 27；`step` 在 279 只填 `prev_sample` 一个字段，本题把 `log_p_x_t_min_1` 也返回。
- 论文是 arXiv:2111.14822（*Vector Quantized Diffusion Model for Text-to-Image Synthesis*）。源码三处点名：86 与 116 的 docstring 写 "section 4.1 of the VQ-Diffusion paper"、365/389 的注释按 "equation (11)" 逐条验算归一化、478 的注释写 "See equation 7"。
- 模块级函数：`index_to_log_onehot` 40-57（`F.one_hot` 54、`permute(0, 2, 1)` 55、`clamp(min=1e-30)` 与 `torch.log` 同在 56）；`gumbel_noised` 60-77（`torch.rand(logits.shape)` 74、双重钳制 75）。
- 表的构建：`alpha_schedules` 80-107、`gamma_schedules` 110-139。构造器 168-212 里 `num_embed` 177、掩码类下标约定 179-180、`bt/btt = (1-a-c)/num_non_mask_classes` 185-187、float64 取 log 189-201、`.float()` 降回 203-208、`timesteps` 212。
- `step` 236-279：`timestep == 0` 捷径 267-268、`q_posterior` 270、加噪 272、`argmax(dim=1)` 274。
- `q_posterior` 281-390：`log_onehot_x_t` 302、`cumulative=True` 的 V 304-306、`cumulative=False` 的 U 308-310、`q = log_p_x_0 - V` 317、`logsumexp` 321、`q - S` 328、`apply_cumulative_transitions(q, t - 1)` 336、`logits = q + U + S` 344、逐列验算的注释 346-389。
- `log_Q_t_transitioning_to_known_class` 392-489：表下标 441-448、保存掩码行 461（注释 450-460）、丢弃 466、`(log_onehot + a).logaddexp(b)` 479（"cheeky trick" 注释 468-478）、整列覆写 482-484、回填 487。
- `apply_cumulative_transitions` 491-516：`c.expand(bsz, 1, num_latent_pixels)` 511、`(q + a).logaddexp(b)` 513、`torch.cat` 514。

## 核心与易错点

- **累积表不是乘出来的。** 100-106：ᾱ 直接在 `alpha_cum_start → alpha_cum_end` 上线性插值，再 `at = att[1:] / att[:-1]` 反推单步值，最后把 `att` 拼回 `[att[1:], 1]`；γ 同理（130-138，用 `1 - ctt` 做比值再取回）。所以「cumulative」只是相对单步表的称呼，实现里是两条同长数组加一个哨兵；行和恒等式 \(\bar a_t + k\bar b_t + \bar c_t = 1\) 来自 186-187 的**定义**，不是任何乘积的性质。
- **三个下标里只有一个是 `t-1`。** V 用 `cumulative=True` 配 `t`（304-306 → 442-444），U 用 `cumulative=False` 配 `t`（308-310 → 446-448），W 走 `apply_cumulative_transitions(q, t - 1)`（336 → 506-508）。累积表被读了两次，一次 `t` 一次 `t-1`；把 V 也写成 `t-1` 等于把分母 \(q(x_t\mid x_0)\) 整体挪一格，比例 \(p(x_0)/q\) 立刻失真。
- **「先保存掩码行再丢弃」只在非累积那次发生。** 450-466：只有 `not cumulative` 才保存 `log_onehot_x_t[:, -1, :].unsqueeze(1)`，随后无条件 `[:, :-1, :]`。累积那次丢掉的是「\(x_0\) 是掩码」那一列——`model_output` 本来就没有这一通道（249-251 的 docstring：初始图像不可能是掩码）；非累积那次丢掉的是 \(q(x_t\mid x_{t-1}=\text{mask})\)，吸收态下命中为 1、否则为 0，取 log 就是 0 与钳制值。这正是题面 \(U[k]=E[k]\) 的来历。
- **钳制值会一路活到最终 logits。** 56 的 `clamp(min=1e-30)` 让 one-hot 非命中项是 `-69.07755278982137` 而不是 `-inf`。在 479 它被 `logaddexp` 盖成 \(\log b\) 看不出差别，但 461/487 那条掩码行原封不动进 344 的加法：题面示例里 `t=1` 未掩码像素的掩码行 \(-68.755728=\log(10^{-30})+\log\bar c_0+\log S\)。
- **`S` 的减与加不是数值技巧。** 317/321/328/336/344 组成「除以 \(q(x_t\mid x_0)\) → 归一 → 乘 \(\bar a_{t-1}\) 加 \(\bar b_{t-1}\) → 把归一化常数乘回」。由于 `logaddexp` 对加法不线性，省掉 328 的减法会让 \(\exp(W)\) 的类别和从 \(1-\bar c\) 变成 \(S(1-\bar c)\)。上游 346-389 用整段注释（掩码列 359/363，非掩码列 379/383）展开验证每列确实是概率，等价于本题的守恒测试：逐像素 `exp(logits).sum(cls) == exp(model_output).sum(cls)`，与 \(t\) 无关。
- **`t=0` 是整段旁路，不是退化公式。** 267-268 直接 `log_p_x_t_min_1 = model_output`，`q_posterior` 完全不进；于是 74 的 `torch.rand(logits.shape)` 也跟着少一行。若真按公式跑 `t=0`，`apply_cumulative_transitions(q, -1)` 会取到 106/138 那个哨兵：\(\bar a_T=1\)、\(\bar c_T=0\)、\(\bar b_T=0\)，取 log 得 \(-\infty\)——Python 的负索引不会报错，只会静默把掩码行变成 `-inf`。这就是本题要求 `0 <= t < T` 且把哨兵排除在校验之外的原因。
- **Gumbel 口径。** 75 是 `-log(-log(u + 1e-30) + 1e-30)`：`noise(0.0) = -4.235229826910111`、`noise(exp(-1)) = 0`。274 的 `argmax(dim=1)` 在平票时取下标最小的类别，题面示例用 `u=exp(-1)` 把噪声精确清零，就是为了让裸 argmax 可手算；平票定序另有专门的用例。
- **上游不做 `T // N` 缩放。** 226 是 `np.arange(0, num_inference_steps)[::-1]`，直接把 `0..N-1` 当表下标用（别的调度器在这里乘 `T//N`）。所以 `set_timesteps(N)` 里 \(N<T\) 时用的是最干净的那一段表。本题把表长视为 `len(at)` 并要求 `t` 是合法下标，不给这个差异留歧义。

## 与上游的差异

- 上游在构造器里按 80-139 自己生成表，并在 203-208 存成 float32；本题吃现成的 `at/ct/att/ctt` 概率表、内部全程 float64。与逐字转写的上游实现（含 float32 降位）对拍 30 组随机配置（\(T\in\{7,11,50,100\}\)、\(C\in\{2,3,17\}\)、\(t\) 覆盖 0 与 \(T-1\)）：argmax 零翻转，logits 最大绝对差 \(6.964\times10^{-6}\)。
- `b` 表不作为入参：185-187 使它由 `(1-a-c)/(C-1)` 完全决定，再给两个参数只会制造可以自相矛盾的输入。
- 噪声：上游 74 自己 `torch.rand`；本题把 `gumbel_uniform` 显式给出，并要求形状随 \(t\) 变（`t=0` 少一行掩码类）。
- 返回值：上游只回 `prev_sample`（279）。本题额外回 `logits`——Gumbel 之后 `argmax` 对 logits 的小扰动不敏感，只对样本判分会漏掉「分布错但采样恰好撞对」这一大类错误。
- 校验：上游允许 `timestep` 是 `torch.Tensor`（239 的类型标注），越界只靠索引自然报错，也不检查 `model_output` 是否真的少一行掩码通道（249-251 只是文档）。本题强制 Python `int`（`bool` 非法）、`0<=t<T`、四张表长度与取值范围，全部抛 `ValueError`。

## 与其他题的分工

- `diff-discrete-ddim-routes` 也是离散状态空间的后验一步，但那是 D3PM 均匀腐蚀核：没有吸收态，后验对 \(x_0\) 有闭式的三路分类（回到预测 token／原地不动／均匀跳）。本题的后验要对全部 \(x_0\) 求和、且靠 `logaddexp` 而不是矩阵乘。
- `diff-amused-topk-remask` 与 `diff-entropy-bound-unmask` 属于「掩码 + 每步提交若干格」的 MaskGIT 家族，考的是**留多少、留哪几个**的调度与置信度排序；本题的掩码由前向过程盖，反向只算一步分布，不涉及提交数量。
- `diff-ddim` / `diff-ddpm-step` / `diff-unipc-pc-step` / `diff-consistency-decoder-2step` 在连续潜变量上做高斯后验；这里的 \(x\) 是类别下标，\(\hat x_0\) 是 `(B, C-1, L)` 的 log 概率。
- `diff-repaint-blend-step` 同样讲「已知区不受网络输出影响」，但那是空间混合法；本题里掩码格的反推概率是吸收态的数学后果（\(q(x_t\mid x_{t-1}=\text{mask})=\delta\)），没有任何显式混合。
- `diff-sigma-grid-construct` / `diff-edm-preconditioning` 那几题管连续 σ 栅格与预条件；本题的调度是 80-139 的「线性插值 + 相邻比值」，两者形状与目的都不同。
