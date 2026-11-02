def solve(token_logits, duration_logits, model_durations,
          blank_id=0, max_symbols=None):
    """NeMo TDT token-and-duration 贪心解码账（NVIDIA/NeMo
    nemo/collections/asr/parts/submodules/transducer_decoding/
    tdt_label_looping.py GreedyTDTInfer.torch_impl 的标量无状态化；
    TDT 论文 arXiv 2304.06795）。

    joint 输出两路独立归一化：token 头（V 含 blank）+ duration 头
    （D 项），model_durations 为 duration 下标 -> 帧数的映射表。
    predictor 无状态化，token_logits/duration_logits 按帧显式入参。

    官方口径要点（torch_impl 逐条钉死）：
      - token = argmax(token 部分)、duration = model_durations[
        argmax(duration 头)]，并列均取首个下标
      - blank 强制 duration >= 1：masked_fill_(durations == 0 &
        blank_mask, 1)——blank 不允许原地停留
      - 非 blank 同样按 duration 跳帧（TDT 与 RNN-T 的本质差异）；
        duration 可把时间推过句末，发射发生在跳帧前仍有效
      - 非 blank duration=0：原地重复发射——官方用 max_symbols 防循
        环：同一时间戳连续非 blank 数达 max_symbols 后强制 blank
        （time_indices += 1，只进 1 帧，忽略 duration）
      - 终止：time >= T 即结束；非 blank + duration 直接出界即结束
        （官方注释 NB: difference between RNN-T and TDT）

    返回 dict（键固定）：tokens / timestamps（跳帧前快照）/
    durations（模型预测原值可含 0）/ final_t（可 > T）/
    forced_blanks。
    域约束：max_symbols is None 且出现非 blank duration=0 ->
    ValueError（无状态 predictor 下必死循环）。
    红线：不得调用 nemo / numba 的任何函数；纯标量循环账。
    """
    raise NotImplementedError
