def solve(sampled_tokens, sampled_probs, sample, timestep, num_inference_steps,
          mask_token_id, threshold, editing_threshold, prompt_mask, committed, random_tokens):
    """块扩散采样的单步定稿决策（diffusers `BlockRefinementScheduler.step` 的离散部分）。

    网络已经对块里每个位置给好候选 token 与置信度；本题只算「这一步把哪些位置定稿、
    哪些位置被改写」，并把结果落回块里。两种模式的配额口径不同，靠 `mask_token_id`
    是否为 None 分流。所有与概率有关的比较都要先把概率降成 float32。

    输入
      sampled_tokens   int64 (B, L)：本步每个位置采到的候选 token id（取值 >= 0）。
      sampled_probs    浮点 (B, L)、有限、取值 [0, 1]：候选 token 的置信度。
      sample           int64 (B, L)：当前的块。
      timestep         Python int t >= 0（0 起的步下标，可以 >= num_inference_steps）。
      num_inference_steps  Python int N >= 1。
      mask_token_id    Python int >= 0 或 None。None -> 均匀破坏模式。
      threshold        有限实数：定稿阈值 tau。
      editing_threshold  None 或有限实数：改写阈值 tau_e。
      prompt_mask      None 或 bool (L,)：True 表示该位是 prompt，不可改写。
      committed        None 或 bool (B, L)：本步之前已定稿的位置（仅均匀模式用）。
      random_tokens    None 或 int64 (B, L)：给未定稿位置的重加噪 token（仅均匀模式用）。

    掩码填充模式（mask_token_id 不为 None）
      active = sample == mask_token_id；其余位置的比较值取 -inf（不是 0）。
      配额表 q = [L//N + 1] * (L % N) + [L // N] * (N - L % N)；quota = q[min(t, N-1)]。
      整块没有任何掩码时本步 transfer 恒空，否则逐行：
        S = {i in active: p_i > threshold}
        |S| >= quota -> transfer = S（全部过阈的都提交，可能多于 quota；quota=0 也走这支）
        |S| <  quota -> 取 active 里 p 降序的前 min(quota, |active|) 名，并列按下标升序
      改写仅在 editing_threshold 不为 None 且 > 0.0 时开启（0.0 与负数关闭）：
        editable = ~active & ~prompt_mask（prompt_mask 为 None 视作全 False）
        editing  = editable & (p_i > editing_threshold) & (sampled_tokens != sample)
      prev_sample = sample，在 transfer | editing 上换成 sampled_tokens；未命中的掩码保持掩码。
      committed 与 random_tokens 在本模式被完全忽略。

    均匀破坏模式（mask_token_id 为 None）
      base = 全 False（t == 0，传入的 committed 一律忽略）或 committed（t > 0）。
      target = ((t + 1) * L + N - 1) // N  # 即 ceil((t+1)*L/N)，累计应定稿数
      needed = max(0, target - base.sum(-1))  逐行
      名次：base 内的位置当 -inf 排最后，其余按 p 降序（并列按下标升序），从 0 起。
      transfer = ~base & ((rank < needed) | (p_i > threshold))
      editing  = base & (sampled_tokens != sample) & (p_i > editing_threshold)
                 —— 只要 editing_threshold 不为 None 就生效，0.0 与负数照样开
      after = base | transfer
      prev_sample = sampled_tokens（transfer 或 editing 命中）
                    sample（after 为真但两条都没命中）
                    random_tokens（after 为假）
      prompt_mask 在本模式被完全忽略。

    输出 (prev_sample, transfer_index, editing_transfer_index)：形状都是 (B, L)，
    类型依次是 int64、bool、bool。

    入参约束（违反任一抛 ValueError）：三个主张量必须同形 (B, L) 且 B >= 1、L >= 1；
    sampled_tokens / sample / random_tokens 是整数张量（bool、浮点非法）且取值 >= 0；
    sampled_probs 是浮点张量、有限、落在 [0, 1]；committed / prompt_mask 非 None 时必须是
    bool 张量且形状分别为 (B, L) / (L,)；timestep、num_inference_steps、mask_token_id 是
    Python int（bool 非法）且 t >= 0、N >= 1、mask_token_id >= 0；threshold 有限实数；
    editing_threshold 为 None 或有限实数；均匀模式必须给 committed 与 random_tokens。
    禁止 import diffusers。
    """
    # TODO
    raise NotImplementedError
