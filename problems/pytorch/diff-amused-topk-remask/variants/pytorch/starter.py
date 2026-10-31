import torch


def solve(
    logits,
    sample,
    drawn_tokens,
    gumbel_uniforms,
    timestep,
    num_inference_steps,
    mask_token_id,
    masking_schedule="cosine",
    temperature=(2.0, 0.0),
    starting_mask_ratio=1.0,
):
    """AMUS 掩码重采样的一步：按「加了 Gumbel 噪声的置信度」把最不确定的 top-k 个位置重新掩掉。

    对齐 diffusers `schedulers/scheduling_amused.py` 里 `AmusedScheduler.step` 的语义
    （MaskGIT 式的迭代 unmask，`mask_by_random_topk` 那一套）。

    入参
        logits           : (batch, seq_len, vocab_size) 浮点张量，模型对当前位置的分布输出；
        sample           : (batch, seq_len) 整型张量，上一步的 token，取值等于 mask_token_id 的位置即「未解出」；
        drawn_tokens     : (batch, seq_len) 整型张量，调用方从 softmax 分布里采好的候选 token（多项采样结果）；
        gumbel_uniforms  : (batch, seq_len) 浮点张量，落在 [0, 1] 的均匀随机数，用于构造 Gumbel 噪声；
        timestep         : 当前步号（Python 整数），调度器从 num_inference_steps-1 递减走到 0；
        num_inference_steps : 推理步数 N（正整数）；
        mask_token_id    : 掩码 token 的编号（Python 整数）；
        masking_schedule : "cosine" 或 "linear"；
        temperature      : 二元组/二元列表 (起点, 终点)，也可以是单个实数；
        starting_mask_ratio : 初始掩码比例，落在 [0, 1]。

    token 与掩码比例
        unknown = (sample == mask_token_id)
        committed = where(unknown, drawn_tokens, sample)              # 即 pred_original_sample
        若 timestep == 0：本步直接收束，prev_sample = committed，不做任何重掩码。
        否则：
            step_idx = N - 1 - timestep                               # timesteps 是 arange(N) 的翻转
            ratio    = (step_idx + 1) / N
            mask_ratio = cos(ratio * pi / 2)   （cosine 型）
                       = 1 - ratio             （linear 型）
            mask_ratio *= starting_mask_ratio
            mask_len = floor(seq_len * mask_ratio)
            mask_len = min(unknown.sum(-1) - 1, mask_len)             # 不许比已掩掉的位置还多
            mask_len = max(1, mask_len)                               # 至少重掩一个
        三条都按 batch 逐行算（ratio 对整批共享），结果取整后是整数。

    置信度与选择
        probs        = softmax(logits, dim=-1)
        p_sel        = probs[i, committed[i]]
        p_sel        = where(unknown, p_sel, finfo(max))               # 已解出的位置置信度置顶，永不被掩
        temps        = linspace(temperature[0], temperature[1], N)    # 单值入参时是 linspace(T, 0.01, N)
        T            = temps[step_idx]
        gumbel       = -log( clamp(-log( clamp(u, 1e-20) ), 1e-20) )   # u 即 gumbel_uniforms
        confidence   = log( clamp(p_sel, 1e-20) ) + T * gumbel
        masking[i]   = confidence[i] < sorted(confidence[i])[mask_len[i]]   # 严格小于第 mask_len 小的值
        prev_sample  = where(masking, mask_token_id, committed)

    返回 (prev_sample, committed, masking, mask_len)：
        prev_sample / committed 是 int64、形状 (batch, seq_len)；masking 是 bool、同形状；
        mask_len 是 int64、形状 (batch, 1)。timestep == 0 时 masking 全 False、mask_len 全 0。
        所有中间量一律用 float64 计算（浮点入参可以传 float32，结果不变）。

    非法输入抛 ValueError：张量类型/形状/ dtype 不符、含 nan / inf、token 越界、
    seq_len < 2、mask_token_id 不在 [0, vocab_size) 内、timestep 不在 [0, N) 内、
    masking_schedule 不是那两个字符串之一、temperature 不是实数或含两个实数的元组/列表、
    starting_mask_ratio 不在 [0, 1] 内。
    禁止调用 diffusers。"""
    raise NotImplementedError
