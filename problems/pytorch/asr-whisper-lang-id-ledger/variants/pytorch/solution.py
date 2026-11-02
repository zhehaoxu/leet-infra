import torch


def solve(logits, lang_tokens, lang_codes, language=None):
    """Whisper 逐窗语言 ID 账（openai/whisper whisper/decoding.py
    detect_language + transcribe.py 首窗策略，v20231105+ 口径）。

    参数
    ----
    logits : (V,) 或 (n_audio, V) 张量——SOT 位置的解码 logits
      （真实实现来自单 token [sot] 前向；此处显式入参，不出模型）
    lang_tokens : 语言 token id 列表（tokenizer.all_language_tokens）
    lang_codes : 语言代码列表，与 lang_tokens 等长同序
      （tokenizer.all_language_codes；probs dict 的键按此顺序插入）
    language : 显式指定语言时 transcribe 跳过检测（首窗策略账）

    官方口径（decoding.py::detect_language 钉死）：

      single = 输入是 2D 才当批；1D 视作单条，unsqueeze(0) 成批
        （官方对 mel：mel.ndim == 2 -> unsqueeze(0)，返回时解开）
      抑制：mask 全 True，语言位置置 False，logits[:, mask] = -inf
        ——非语言位置一律 -inf（不是 0；softmax 后严格为 0）
      language_tokens = logits.argmax(dim=-1)（平分取首个下标）
      language_token_probs = logits.softmax(dim=-1)——**抑制后 softmax
        即在语言集合上重归一化**（raw softmax 的语言位概率 ≠ 此值）；
        exp(-inf) = 0，probs dict 只收语言位（键 = lang_codes 按序插入）
      transcribe 侧：language=None 才检测，且只跑**首个 30s 窗**
        （pad_or_trim(mel, N_FRAMES)），其余窗沿用；选定 =
        max(probs, key=probs.get)——Python max 平分取**首个最大键**
        （即 lang_codes 插入序）
      返回：single 时 tokens 标量 + probs 单 dict；批时张量 + dict 列表
      tokenizer 无语言 token / 空表 -> ValueError（官方
        "This model doesn't have language tokens so it can't perform
        lang id"）

    需要产出的账（键固定）：
      detected        是否执行了检测（language 显式时 False）
      language        选定语言代码（language 显式时透传；检测时 = top）
      windows_consumed 执行检测的窗下标列表（首窗策略：[0] 或 []）
      tokens          每条音频的语言 token id（single 时标量）
      probs           每条音频 {code: prob}（single 时单 dict）
      prob_sum        每条 probs 值之和（重归一化验证，应 = 1）
      top             每条 max(probs, key=probs.get) 的键（平分取首）
      raw_top_prob    每条 top 语言位**未重归一化**的 softmax 概率
                      （对原始 logits 全词表 softmax，作重归一化对照）

    红线：不得调用 whisper / transformers 的任何函数；仅 torch/numpy。
    """
    if not lang_tokens or not lang_codes:
        raise ValueError("This model doesn't have language tokens "
                         "so it can't perform lang id")
    if len(lang_tokens) != len(lang_codes):
        raise ValueError("lang_tokens/lang_codes length mismatch")

    out = {"detected": language is None}
    if language is not None:
        out.update({"language": language, "windows_consumed": [],
                    "tokens": None, "probs": None, "prob_sum": None,
                    "top": None, "raw_top_prob": None})
        return out

    single = logits.ndim == 1
    if single:
        batch = logits.unsqueeze(0)
    elif logits.ndim == 2:
        batch = logits
    else:
        raise ValueError("logits must be 1D or 2D")

    V = batch.shape[-1]
    mask = torch.ones(V, dtype=torch.bool)
    mask[list(lang_tokens)] = False
    suppressed = batch.double().clone()
    suppressed[:, mask] = -float("inf")

    tokens = suppressed.argmax(dim=-1)
    full_probs = torch.softmax(suppressed, dim=-1)
    raw_full = torch.softmax(batch.double(), dim=-1)

    all_probs = []
    sums = []
    tops = []
    raw_tops = []
    for i in range(batch.shape[0]):
        d = {}
        for j, c in zip(lang_tokens, lang_codes):
            d[c] = float(full_probs[i, j])
        all_probs.append(d)
        sums.append(sum(d.values()))
        top = max(d, key=d.get)
        tops.append(top)
        top_idx = lang_codes.index(top)
        raw_tops.append(float(raw_full[i, lang_tokens[top_idx]]))

    out.update({
        "language": tops[0],
        "windows_consumed": [0],
        "tokens": tokens[0].item() if single else tokens.tolist(),
        "probs": all_probs[0] if single else all_probs,
        "prob_sum": sums[0] if single else sums,
        "top": tops[0] if single else tops,
        "raw_top_prob": raw_tops[0] if single else raw_tops,
    })
    return out
