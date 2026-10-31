# 解题说明 · AMUS 加噪置信度重掩码

## 出处

- `huggingface/diffusers` 的 `src/diffusers/schedulers/scheduling_amused.py`（v0.42.1）：`AmusedScheduler.step`（第 124–191 行）是本课原型，配套的 `mask_by_random_topk`（第 31–58 行）、`gumbel_noise`（第 12–28 行）与 `set_timesteps`（第 111–122 行）各管一段。类 docstring 自述「A scheduler for masked token generation as used in [`AmusedPipeline`]」。
- 「按置信度逐步 unmask」是 MaskGIT 一系采样器的共同骨架（MaskGIT，arXiv:2202.04200）；AMUS 这一支的特色是把选择判据从纯概率换成 \(\log p + T\cdot\text{Gumbel}\)，让高温的前几步保留随机性，避免早早锁死。
- 源码有两处现采随机：候选 token 走 `torch.multinomial`（第 149 行），Gumbel 噪声走 `gumbel_noise` 内部的 `uniform_(0, 1)`。本题把两者都改成显式入参 `drawn_tokens` / `gumbel_uniforms`，把随机性摘出契约，只留下确定性的调度与比较。

## 核心与易错点

**`step_idx` 与 `timestep` 反号。** 源码 `set_timesteps` 把 `self.timesteps` 存成 `arange(N).flip(0)`，`step` 里用 `(self.timesteps == timestep).nonzero()` 反查下标，于是 `step_idx = N - 1 - timestep`、`ratio = (step_idx + 1) / N`。写成 `ratio = (timestep + 1) / N` 会让整条掩码调度倒过来：末步埋得最多、首步几乎不埋，而 token 序列看着仍然「在变」，很容易自检漏掉。测试用 `check_schedule_tables` 逐 step 比对两张表（cosine 与 linear 各一），把方向钉死。

**两道钳位的顺序与各自的含义。** `mask_len = floor(S * mask_ratio)` 之后先 `min(sum(unknown) - 1, .)`（源码注释：不要埋得比已经掩着的还多），再 `max(1, .)`（至少埋一个）。前者保证每步**净减少**至少一个掩码——这也意味着 `mask_len` 上界是「未知位置数减一」，而不是未知位置数；后者是无条件的，连「整行已经没有未知位置」这种退化输入也会被埋回 1 个 token，此时被埋的确实是**已经提交过的位置**。`check_clamped_by_known_count` 与 `check_no_unknown_still_masks_one` 分别抓这两条。

**哨兵替换发生在取对数之前。** 源码是 `selected_probs = where(unknown, selected_probs, finfo.max)` 之后才进 `mask_by_random_topk`，而后者内部再 `log(clamp(p, 1e-20))`。所以已知位置的置信度是 \(\ln(\texttt{finfo.max}) + T g \approx 709.78 + T g\)，仍然带着自己的噪声项；把它当成常数 709.78（先 log 再 where）在这道题的取值范围内不会翻转结果，但会让「已知位置恰是门槛」这类边界用例的语义漂移，契约按源码顺序写死。

**门槛是升序下标 `mask_len`，判据是严格小于。** `cut_off = sorted_confidence[mask_len]` 取的是第 `mask_len + 1` 小的值，比较用 `<` 而非 `<=`。因此无并列时恰埋 `mask_len` 个；一旦并列（例如三个位置的 `gumbel_uniforms` 相同且门槛正落在它们身上，或整行都已知且噪声一致），埋回个数会**少于** `mask_len`，极端情况一个都不埋。测试里 `check_ties_mask_fewer` 与 `check_all_tied_nothing_masked` 分别压出「少于」与「一个都不埋」这两条退化路径。

**两处 clamp 都要写。** `-log(u)` 在 \(u = 0\) 处发散、`-log(-log(u))` 在 \(u = 1\) 处又发散一次，源码用两次 `clamp(1e-20)` 把它们钉成有限：\(g \in [-3.829, 46.05]\)。概率侧同理，`logits` 相差 200 时选中概率下溢成 0，按 \(10^{-20}\) 计入。`check_uniform_clamp` 用 `u = 0 / 0.5 / 1` 三点与一对极端 logits 把两侧都压出来。

**`temperature` 有两种写法且与 `ratio` 共用索引。** 元组/列表 → `linspace(t0, t1, N)`；单个数 → `linspace(T, 0.01, N)`（终点是 0.01 而不是 0）。取的是 `temperatures[step_idx]`，不是 `temperatures[timestep]`。\(T = 0\) 时置信度退化成纯 \(\log p\)，「埋概率最低的 `mask_len` 个」这条可手算的性质就是靠它来验证的（`check_temperature_zero_picks_lowest_confidence`）。

**`timestep == 0` 只短路重掩码，不短路提交。** 候选 token 照样顶上未知位置，`committed` 仍是要返回的量；只是不再埋、`masking` 全 False、`mask_len` 全 0。把整个函数写成「末步返回原 `sample`」会漏掉这一步的提交。

## 与上游的差异

- **计算精度**：源码在 CPU 上以 float32 走完 softmax 与置信度，`torch.finfo(torch.float32).max` 取对数是 88.72；本题契约钉死 float64，哨兵项是 709.78。在 \(|T| \le 8\) 下两者给出同一套掩码集合（哨兵位与未知位的最小间隔远大于浮点差），但 \(|T|\) 再大就可能出现「已知位置被哨兵压过」的情形——契约因此限制 \(|T| \le 8\)；按 \(g\) 的上下界 \([-3.829, 46.05]\) 精算，安全上界约 14.2，取 8 是留余量。
- **只支持 2-D 输入**：源码的 `two_dim_input` 分支（`(B, S, H, W)` 的图像 token 先 reshape + permute 成 `(B, H*W, S)`）与 `add_noise`（按 `mask_ratio` 掷均匀骰子加掩码）都不在本题范围。
- **非法输入语义**：源码对「`timestep` 不在 `self.timesteps` 里」不报错，`nonzero()` 返回空张量后一路静默产出错误 shape；`S = 1` 时会因 `gather` 的索引越界直接报错。本题统一改成 `ValueError`，并显式要求 \(S \ge 2\)、`timestep` 落在 \([0, N)\)。
- **返回内容**：`return_dict=False` 时源码只给 `(prev_sample, pred_original_sample)`；本题恒定返回四元组，把 `masking` 与 `mask_len` 也作为可观测中间量交出，形状沿用源码的 `(B, 1)`（带尾单维度）而不是压平成 `(B,)`。
- 采样侧的 `multinomial` 与 `gumbel_noise` 被显式入参替代；`starting_mask_ratio` 保留为参数（默认 1.0），因为它直接参与 `mask_len` 的算式。

## 与其他题的分工

- `diff-entropy-bound-unmask`：同是离散扩散的「这一步提交哪些位置」，但那题服务 uniform-corruption（没有 MASK token），判据是温度缩放后的**熵前缀和**，未接收的位置换成**均匀随机 token**；本题服务 MASK-token 家族，判据是**加噪对数概率的 top-k**，未保留的位置换回 `mask_token_id`，并且额外管「埋多少个」的调度与两道钳位。
- `diff-discrete-ddim-routes`：D3PM 的离散转移路由（吸收/均匀腐蚀的转移矩阵），算的是分布本身而非提交集合。
- `diff-repaint-jump-schedule`：连续图像空间里按跳步表决定「什么时候重来」；本题在离散 token 空间按置信度决定「重来哪些位置」。
- `dt-unigram-sample` / `dt-bpe-encode`：分词器侧的离散选择与编码，没有去噪步序，也没有随步退火的温度。
