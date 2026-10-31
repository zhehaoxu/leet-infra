# Notes: 块精化采样的一步与两种配额口径

## 出处

- diffusers v0.42.1 `src/diffusers/schedulers/scheduling_block_refinement.py`（全文 520 行），类 `BlockRefinementScheduler` 在 54-518（`add_noise` 是类里最后一个方法），`order = 1` 在 66，`__all__` 在 520。
- 这个文件**没有点名任何论文**：112 只留了一行 `# --- SAR sampling utilities ---`（SAR = semi-autoregressive），216 与 264 两处把均匀破坏模式写作 "matching DiffusionGemma's block refinement sampler"。同一套「块内并行、按置信度逐步定稿」的非因果块扩散机制发表在 BD3-LM（arXiv:2503.09573，ICLR 2025，`kuleshov-group/bd3lms`），本题把它当**机制出处**引用，源码本身不含该链接。
- 配置默认值在 68-81：`block_length=32`（71）、`num_inference_steps=32`（72）、`threshold=0.95`（73）、`editing_threshold=None`（74）、`minimal_topk=1`（75）；`self.timesteps = torch.arange(N-1, -1, -1)` 在 78 与 96，`_transfer_schedule`/`_committed` 两个状态在 79-81（80 的注释写明 committed "reset at the start of each block"）。
- 配额表：`get_num_transfer_tokens` 在 102-110，`base = L // N`、`remainder = L % N`（106-107）、`torch.full` 后 `out[:remainder] += 1`（108-109），余数**前置**；`num_inference_steps <= 0` 在 104-105 返回空张量。`set_timesteps` 在 83-100，对 \(N\le0\)（89-90）与 \(L\le0\)（93-94）抛 `ValueError`，并在 100 清空 `_committed`。
- `step` 在 181-360：`threshold`/`editing_threshold`/`minimal_topk` 的 config 兜底在 239-244，采样在 247-255，`step_index` 的取整在 259-262。均匀破坏模式 264-302（重置条件 266-267、float32 转换 269、累计配额注释 271、target 273、`clamp(min=0)` 274、`masked_fill(-inf)` 276、双重 `argsort` 277、`transfer_index` 278、editing 分支 280-284、`prev_sample` 286、状态更新 287、`torch.randint` 288-290、重加噪 291）。掩码填充模式 304-349（`active_block` 304、`masks_remaining` 305、`clamped_step` 310、`num_to_transfer` 311、`-inf` 填充 313-317、逐行循环 319-327、`editing_enabled` 330、`prompt_mask` 兜底 333-334、`editable` 335、编辑置信度 336-343、落盘 346-349）。
- 循环外还有三个辅助：`check_eos_finished`（362-410，静态方法，EOS 落定后收尾）、`check_block_should_continue`（412-454）、`add_noise`（456-518，逐块随机比例的补集双视图）。本题都不考。

## 核心与易错点

- **两套配额口径不是一套的两个写法。** 掩码模式吃 310-311 的**每步定量** \(q_{\min(t,N-1)}\)；均匀模式吃 273 的**累计目标** \(\lceil (t+1)L/N\rceil\)，再在 274 减掉已提交数。\(L=10,N=6\)：每步表 \([2,2,2,2,1,1]\)，前缀和 \([2,4,6,8,9,10]\)，累计目标却是 \([2,4,5,7,9,10]\)（增量 \([2,2,1,2,2,1]\)）。只有 \(N\mid L\) 时两者才一致。
- **`high_conf.sum() >= num_to_transfer` 是 `>=`，命中后提交的是全部过阈位置**（321-322），数量可以远超配额；配额为 0（\(N>L\) 的后段，`base=0` 且 `remainder` 已用完）也照样进这条分支，此时「有一个过阈就提交一个」。反过来走 top-k 时，`k = min(num_to_transfer, active.sum())`（324）把越界压回来，`k=0` 直接跳过（325）——这一行既可能是「配额没用满」也可能是「该行压根没掩码」。
- **`masks_remaining` 是全局判断**（305，`active_block.any()` 不带 batch 维），但它只决定要不要进 309 那个 `if`；进入之后逐行仍各自算。整块一个掩码都没有 → 本步 `transfer` 恒空，哪怕置信度全都过阈；只有某行没有掩码 → 只空那一行。
- **`editing_threshold` 的开关在两个模式里语义相反。** 330 是 `is not None and > 0.0`：`0.0` 与负数**关闭**编辑（63 行的类文档字符串明说 "`None`, `0.0`, or negative disables editing"）；281 只判 `is not None`：`0.0` 与负数**开启**编辑，此时 `confidence > 0.0` 几乎恒真，编辑集合退化成 `committed & (tokens != sample)`。
- **均匀模式的 `committed` 是「本步之前」的状态，且 `t == 0` 强制清零**（266-267 的条件里还有形状不符也重置）。排名用 276 的 `masked_fill(-inf)` 把已提交位置踢到最后，277 的 `argsort(..., descending=True).argsort(...)` 得到名次；因此 278 的 `transfer_index` 必须再乘一道 `~committed`——只写 `ranks < needed` 会把已提交位也算进去。
- **278 的并集里阈值项没有被配额截断**：均匀模式同样是「配额 ∪ 过阈」，与掩码模式的 `>=` 分支同源，但它在同一步里可以同时命中两条，`needed` 只约束排名那半边。
- **最后那次重加噪发生在更新后的状态上**（287 先 `self._committed = committed | transfer_index`，291 才用它做 `torch.where`）。所以本步刚定稿的位置不会被 `randint` 冲掉；被编辑的位置也不会（它本来就在 `committed` 里）。写成 `where(committed, ...)` 会把刚提交的一整批又刷成随机 token。
- **`prompt_mask` 只在掩码模式的编辑支路被读**（333-335），均匀模式从头到尾不看它；反过来 `committed`/`random_tokens` 在掩码模式不被读。两个模式的忽略项正好互补。
- **所有比较都在 float32 里**：269、315、338 三处都有 `.to(dtype=torch.float32)`。给一个 float64 的 `sampled_probs = [0.9500000000000001, 0.96, 0.5, 0.4]` 与 `threshold = 0.95`：留在 float64 比，过阈是 \(\{0,1\}\)；先降到 float32（0.9500000000000001 → \(0.949999988079071\)，与 float32 化的阈值恰好同一个数）再比，过阈只剩 \(\{1\}\)。\(L=4,N=4\)（每步配额 1）时两条都走 `>=` 分支，但前者多提交一个位置，实测差在 `transfer_index[0][0]`。
- **未选中位置的比较值是 \(-\infty\) 而不是 0**（316、339）。用 0 填充会让 `threshold < 0` 的入参把非活动位也算成过阈；`-inf` 还保证它们在任何 top-k 里排在所有真实概率之后。
- **并列名次按下标升序取**。上游 326 用的是 `torch.topk`，其并列顺序未文档化；实测在 CPU float32 上，`(v.argsort(descending=True).argsort(dim=-1) < k)` 与 `topk(v, k).indices` 给出的**集合**在 4914 个随机步（\(L\in\{5..16\}\)、\(N\in\{2..7\}\)、三种阈值、含 `-inf` 与精确并列）里逐个相同，本题据此把并列规则写死成「下标小者优先」。
- **`minimal_topk` 是死参数**：243-244 从 config 取出来以后，`step` 全文再没引用过它（grep 只命中 75、194、230、243、244）。任何「至少提交 k 个」的保底逻辑都不存在，`quota` 与 top-k 都可能给出空行。
- **`timesteps` 是降序数组，`step` 却把入参当下标用**（78/96 造 `arange(N-1,-1,-1)`，259-262 直接 `int(timestep)`，266 又用 `step_index == 0` 判「块开始」。）两处约定互相矛盾：真按降序数组喂参数，`committed` 的清零会发生在块的**最后**一步。本题按 209 行的 docstring（"Current step index within the block's refinement schedule"）取「0 起的步下标」这一解释。

## 与上游的差异

- 上游 `step` 的第一个入参是 `model_output`（\((B,L,V)\) 的 logits），并在 247-255 内部做温度/top-k/top-p/`multinomial` 采样。本题把采样结果 `sampled_tokens`、`sampled_probs` 直接给出，logits、温度、`top_p`、`top_k`、`sampling_method` 一律不进题面。顺带一条：162-166 的贪心支路（`temperature == 0.0` 或 `not use_multinomial`）压根不调用 `_top_k_filtering`/`_top_p_filtering`，即贪心时 `top_p`/`top_k` 是无效的。
- 上游 288-290 自己 `torch.randint` 生成重加噪 token；本题把它当入参 `random_tokens` 给出，`generator` 不进题面。
- 上游把 `_committed`、`_transfer_schedule` 藏在实例状态里，靠 `set_timesteps`（100）与 `step_index == 0`（266）两处隐式重置；本题把 `committed` 摊成显式入参，只在 `t == 0` 时忽略它，单步无状态。
- 上游配额表来自 `self._transfer_schedule`，为 `None` 时（309 的第二个条件，即从未调用 `set_timesteps`）整步 `transfer` 为空；本题按 \(L,N\) 现算，并要求 \(N\ge1\)，因此 `num_inference_steps <= 0` 返回空张量那条（104-105）不会出现。
- 上游不对 `sampled_probs` 的值域、张量形状、`threshold` 的有限性做任何校验（`step` 里没有一条 `raise`，靠 `sample.shape` 与 `mask_token_id` 隐式成立）；本题要求形状一致、\(p\in[0,1]\)、有限、整数入参是 Python int（bool 非法），违反抛 `ValueError`。
- 上游返回六元组/`BlockRefinementSchedulerOutput`（293-302、351-360），把 `sampled_tokens`、`sampled_probs`、`pred_logits` 原样回传用于下一步自条件；本题只回三个决策量。
- 上游 `prompt_mask` 允许广播、`mask_token_id` 未校验非负；本题要求 `prompt_mask` 恰好 \((L,)\)。
- 题面不含 `check_eos_finished`（362-410）、`check_block_should_continue`（412-454）与 `add_noise`（456-518）：前两个是采样循环的收尾判据（EOS 落定、后处理步数上限），第三个是训练侧的前向掩码构造，都不属于单步提交决策。

## 与其他题的分工

- `diff-amused-topk-remask` 同是离散侧「每步提交一部分、其余重掩码」，但那里的配额来自 `MaskScheduler.sample_topk` 的概率质量阈值与 `step` 里的重新掩码；本题的配额是块长平摊，且完全没有概率分布的截断，只有名次与两个阈值。
- `diff-entropy-bound-unmask` 决定的是掩码 LM「本步解开哪几个位置」，判据是熵/置信度排序 + 长度表；那是一道单口径的排序题，本题的重点恰在**两个模式的口径不同**。
- `diff-vq-diffusion-posterior` 与 `diff-ddpm-step` 等连续/潜变量调度题管的是更新式里的系数；本题没有任何系数，纯粹是离散决策（返回 int64 与 bool，逐位比对、零容差）。
- `dt-*` 系列（如 `dt-chat-template`、`dt-bytelevel-pretokenize`）管分词与模板装配，与「哪些位置在本步定稿」无关。
- 合集内暂无另一道题涉及「半自回归块并行解码」这一族；后续若补 `check_eos_finished`/`add_noise`（同一文件的 362-410 与 456-518），应单独成题，不要塞进本题。
