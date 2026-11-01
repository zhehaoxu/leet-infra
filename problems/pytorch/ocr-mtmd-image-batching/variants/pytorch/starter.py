def solve(chunks, n_batch, logits_last=False, n_past=0):
    """llama.cpp mtmd 分批调度账（mtmd_helper_eval_chunks / eval_chunk_single /
    decode_image_chunk 口径）。

    输入
    ----
    chunks      : [{"type": "text"|"image", "n_tokens": int}, ...] 输入 chunk 序列
                  （token id 与 embedding 数值不入题，只考调度账）
    n_batch     : 单批 token 上限（正 int，官方 GGML_ASSERT(n_batch > 0)）
    logits_last : 是否在整段输入的最后一个 token 上取 logits
    n_past      : 起始位置

    记账（题面钉死，与官方一致）：
      1. 空 chunk 序列：直接返回，无批、位置不动（官方 "no chunks to eval"）；
      2. text chunk：while 循环按 n_batch 切批（每批 <= n_batch），位置**逐
         token 推进**——批 k 的 pos 区间 [n_past + 已发, +批大小)；logits 只在
         「本 chunk 是最后一个 chunk 且 logits_last 且本批是 chunk 内最后一批」
         时设在批内最后一个 token 上；
      3. image chunk：先编码后分批解码，n_img_batches = ceil(n_tokens / n_batch)，
         批 k 的 pos 区间 [n_past + k*n_batch, +min(n_batch, 剩余))——位置是
         全 chunk 一次设 pos_0 + i 后按视图偏移取用；**chunk 解码完后位置一次
         推进 n_tokens**（非 M-RoPE 下 n_pos == n_tokens）；**image chunk 分支
         不处理 logits_last**——最后一个 chunk 是 image 时无 logits（官方 quirk）；
      4. logits_pos = 最后一次标记 logits 的位置（未标记则 None）。

    返回
    ----
    dict：
      batches     list[dict]  调度序批账：{chunk_index, kind, n_tokens, pos_start, logits}
      new_n_past  int         位置推进总量（= n_past 入参 + 各 chunk 推进）
      logits_pos  int|None    取 logits 的位置
    """
    raise NotImplementedError
