import torch


def solve(token_logits, duration_logits, model_durations,
          blank_id=0, max_symbols=None):
    """NeMo TDT token-and-duration 贪心解码账（NVIDIA/NeMo
    nemo/collections/asr/parts/submodules/transducer_decoding/
    tdt_label_looping.py GreedyTDTInfer.torch_impl 的标量无状态化；
    TDT 论文 arXiv 2304.06795）。

    joint 输出两路独立归一化：token 头（V 含 blank）+ duration 头
    （D 项），model_durations 为 duration 下标 -> 帧数的映射表。
    predictor 无状态化（官方 conformer_tdt_bpe_stateless 配置同构），
    故 token_logits/duration_logits 按帧显式入参。

    官方口径（torch_impl 逐条钉死）：

      token = argmax(token 部分)（并列取首个下标）
      duration = model_durations[argmax(duration 头)]（并列取首个）
      **blank 强制 duration >= 1**：durations.masked_fill_(
        durations == 0 & blank_mask, 1)——blank 不允许原地停留
      **非 blank 同样按 duration 跳帧**（TDT 与 RNN-T 的本质差异：
        发射后直接跳 duration 帧，无需 blank 收尾；duration 可把
        时间推过句末——发射发生在跳帧前，仍有效）
      发射账：token 发射于跳帧前的时间戳（time_indices_current_labels
        在 advance 前快照）
      **非 blank duration=0**：原地重复发射（label loop）——官方用
        max_symbols 防循环：同一时间戳连续发射的非 blank 数达到
        max_symbols 后强制 blank（time_indices += 1，**忽略 duration
        只进 1 帧**）。本题域约束：max_symbols is None 且出现非
        blank duration=0 -> ValueError（无状态 predictor 下必死循环）
      终止：time >= encoder_output_length（active_mask）；不存在
        RNN-T 的"发射后需 blank 才能结束"——非 blank+duration 直接
        出界即结束（官方注释 NB: difference between RNN-T and TDT）

    参数
    ----
    token_logits : (T, V) 张量/嵌套 list（V 含 blank）
    duration_logits : (T, D) 张量/嵌套 list
    model_durations : list[int]，长度 D，非负
    blank_id : blank 下标（默认 0）
    max_symbols : 同一时间戳最大连续非 blank 发射数（None = 官方缺省
      无防护；本题配合域约束）

    需要产出的账（键固定）：
      tokens         发射的非 blank token 序列
      timestamps     各 token 的时间戳（跳帧前快照）
      durations      各 token 的 duration（模型预测原值，可含 0）
      final_t        终态时间（可 > T——末 token duration 跳出句末）
      forced_blanks  max_symbols 强制 blank 次数

    红线：不得调用 nemo / numba 的任何函数；纯标量循环账。
    """
    if isinstance(token_logits, torch.Tensor):
        token_logits = token_logits.tolist()
    if isinstance(duration_logits, torch.Tensor):
        duration_logits = duration_logits.tolist()
    T = len(token_logits)
    if T == 0 or len(duration_logits) != T:
        raise ValueError("token_logits/duration_logits length mismatch")
    if any(d < 0 for d in model_durations):
        raise ValueError("model_durations must be non-negative")
    V = len(token_logits[0])
    D = len(model_durations)
    if any(len(r) != V for r in token_logits) or \
            any(len(r) != D for r in duration_logits):
        raise ValueError("logits width mismatch")

    def argmax_row(row):
        best = 0
        for i in range(1, len(row)):
            if row[i] > row[best]:
                best = i
        return best

    tokens = []
    timestamps = []
    durations_out = []
    t = 0
    last_nb_t = -1
    last_nb_lasts = 0
    forced_blanks = 0
    max_steps = 100 * (T + 1)       # 防御步数上限（官方无此防护，
    steps = 0                        # 仅病理输入会触发，见 notes）
    while t < T:
        steps += 1
        if steps > max_steps:
            raise RuntimeError("step limit exceeded (official greedy "
                               "would loop forever on this input)")
        tok = argmax_row(token_logits[t])
        d_idx = argmax_row(duration_logits[t])
        dur = model_durations[d_idx]
        if tok == blank_id:
            if dur == 0:
                dur = 1                     # blank 强制 >= 1
        else:
            if dur == 0 and max_symbols is None:
                raise ValueError("non-blank zero duration would loop; "
                                 "set max_symbols")
            tokens.append(tok)
            timestamps.append(t)
            durations_out.append(dur)
            if t == last_nb_t:
                last_nb_lasts += 1
            else:
                last_nb_lasts = 1
                last_nb_t = t
            if max_symbols is not None and last_nb_lasts >= max_symbols:
                t += 1                      # 强制 blank：只进 1 帧
                forced_blanks += 1
                continue
        t += dur
    return {"tokens": tokens, "timestamps": timestamps,
            "durations": durations_out, "final_t": t,
            "forced_blanks": forced_blanks}
