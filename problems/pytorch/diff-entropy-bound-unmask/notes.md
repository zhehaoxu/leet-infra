# 解题说明 · 熵界解掩码

## 出处

- `huggingface/diffusers` 的 `src/diffusers/schedulers/scheduling_entropy_bound.py`：`EntropyBoundScheduler.step`（第 118–182 行）是本课的原型。源码注释写明它服务的是 uniform-corruption 的离散扩散——没有 MASK token，未接收的位置直接换均匀随机 token。
- 论文出处见源码 docstring：*Accelerated Sampling from Masked Diffusion Models via Entropy Bounded Unmasking*（arXiv:2505.24857）。论文给出的界是「被接收 token 之间的联合互信息上界」；由于排序后每个位置只需减去自身那一份熵，工程实现就落成了 `cum - H_self <= bound` 这个逐位判据。
- 源码里候选 token 由 `torch.multinomial` 现场采样（还返回 `sampled_probs`、`pred_logits`）；本题把候选 token 与随机 token 都作为**输入**注入，把随机性从契约里摘掉，只保留确定性部分。

## 核心与易错点

**前缀和要「减去自己」。** 源码写的是 `cumulative_entropy - sorted_token_entropy <= entropy_bound`：`cumsum` 含自身，所以要减掉自身才等价于「比自己熵更低的所有位置之和」。写成 `cumsum <= bound` 会让 \(B=0\) 一个都接收不到（第一个位置的 cum 就等于它自己的熵），且整体少接收一个位置——测试里 `B=0` 只接收最低熵位置这一条专抓这个。

**温度作用在 logits 上，熵也用缩放后的分布算。** 源码先 `model_output = model_output / temperature`，再用同一个张量同时做采样与熵。熵对温度不是不变量（均匀分布除外），若用原始 logits 算熵，温度退火就完全失效。测试用 `t_max=t_min=1.0` 把温度钉成 1，再与手算的 \(-\sum p\log p\) 比对，正好把「缩放是否作用到熵」隔离出来。

**退火方向：`timestep` 越小越热。** `fraction = (N - timestep) / N`，`timestep=0` 时 fraction 为 1 → 温度 \(t_{\max}\)；末步只是逼近 \(t_{\min}\)，实际是 \(t_{\min}+(t_{\max}-t_{\min})/N\)。把 timestep 当成「剩余步数」或写成 `timestep / N`，温度曲线会整个反向。

**排序的并列语义要固定。** `torch.sort` 默认不保证稳定，并列熵的先后是未定义的。本题契约把并列规定为「下标小者优先」，实现用 `stable=True`；oracle 用 `(熵, 下标)` 作为排序键对齐同一语义。否则同分布的位置会在 `B` 恰好卡在边界时随机互换接收资格。

**接收顺序与输出顺序是两套下标。** 排序下标 `sorted_indices` 只用于「按序判定 + 散回」；`prev_sample`、`accepted`、`token_entropy` 的最后一维都必须回到**原始位置**。只对 `sorted_accepted` 做 `where` 会得到按熵排列的错位 token 块。

**熵的数值稳定写法。** 直接 `p * log(p)` 会在极端 logits 下产生 `0 * -inf = nan`；用 `log_norm = logsumexp(z)`、`log_p = z - log_norm`、`H = -Σ exp(log_p) * log_p`，溢出项自然消失。`Categorical(logits).entropy()` 内部就是这条路径。

**类型约定。** 接收掩码是 bool，token 是 int64（源码是 LongTensor），熵是 float32；内部熵用 float64 算，避免 float32 在跨位置的 cumsum 上累积舍入。

## 与其他题的分工

- `diff-ddpm-step` / `diff-ddim`：连续高斯空间的一步更新，状态是浮点张量；本题状态是离散 token id，没有方差项，靠熵来决定「这一步能定几个位置」。
- `diff-repaint-jump-schedule`：同样回答「下一步走哪里 / 哪些位置重来」，但那是图像空间里按时间步编排的确定性表；本题按**当前分布熵**动态决定接收集合。
- `dt-unigram-sample`：同样是「从分布里采离散 token」，但目的是分词器训练/编码；本题是解码期的一次批量提交。
- `diff-v-prediction-convert`：连续扩散的预测目标互转，与本课的离散接收判据没有交集。
