def solve(logits, lang_tokens, lang_codes, language=None):
    """Whisper 逐窗语言 ID 账（decoding.py::detect_language +
    transcribe.py 首窗策略）。

    logits: (V,) 单条或 (n_audio, V) 批的 SOT 位置 logits（显式入参）。
    官方口径：
      - 1D 视作单条 unsqueeze(0)，返回标量 + 单 dict；2D 批返回列表
      - mask 全 True、语言位置 False，非语言位 logits = -inf（非 0）
      - tokens = 抑制后 argmax（平分取首个下标）
      - probs = 抑制后 softmax —— 语言集合上重归一化（≠ raw softmax）
        dict 键按 lang_codes 插入序
      - transcribe: language 显式 -> 跳过检测；否则只跑首窗 [0]，
        选定 = max(probs, key=probs.get)（平分取首个插入键）
      - 空语言表 / 长度不齐 -> ValueError
    返回 dict：detected / language / windows_consumed / tokens / probs /
    prob_sum / top / raw_top_prob（raw 为未重归一化对照）
    红线：不得调用 whisper / transformers。
    """
    raise NotImplementedError
