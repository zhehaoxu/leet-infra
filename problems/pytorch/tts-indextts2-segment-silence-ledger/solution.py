"""IndexTTS-2 推理前后处理账（order 835）。

回源：index-tts/index-tts (main)
  indextts/utils/front.py    TextTokenizer.split_segments_by_token / split_segments
  indextts/infer_v2.py       remove_long_silence / 生成循环 code_lens / target_lengths

判题约定：单入口 solve(mode, **kw)，3 个 mode：
  split_segments       文本 token 按标点优先级递归分段 + 超长硬切 + 合并状态机
  remove_long_silence  静音 token 连续收缩（最多留 10）+ batch 级 isfix 重拼 + clip
  mel_length_ledger    生成 codes 截到 stop / 批内切齐 / target_lengths×1.72 帧账

不进判题域（避免与既有题重复 / 需真模型）：真实 tokenizer / BPE、GPT 采样与
emovec、s2mel length_regulator 的帧扩展（= FastSpeech2 length-reg）、BigVGAN、
真音频 IO / mel / 波形、训练式时长预测器、latent 插值。
"""

PUNCTUATION_MARKS_TOKENS = [".", "!", "?", "▁.", "▁?", "▁..."]
DEFAULT_MAX_SEG = 120
DEFAULT_RATIO = 1.72
DEFAULT_SILENT_TOKEN = 52
DEFAULT_MAX_CONSECUTIVE = 30
DEFAULT_MAX_MEL_TOKENS = 1500


def _split(tokenized_str, split_tokens, max_seg, quick):
    """返回 (merged_segments, pre_merge_segments, total_token, merge_flags)。"""
    if len(tokenized_str) == 0:
        return [], [], 0, []
    segments = []
    cur = []
    cur_len = 0
    for i in range(len(tokenized_str)):
        token = tokenized_str[i]
        cur.append(token)
        cur_len += 1
        if not ("," in split_tokens or "▁," in split_tokens) and \
                ("," in cur or "▁," in cur):
            sub, _, _, _ = _split(cur, [",", "▁,"], max_seg, quick)
        elif "-" not in split_tokens and "-" in cur:
            sub, _, _, _ = _split(cur, ["-"], max_seg, quick)
        elif cur_len <= max_seg:
            if token in split_tokens and cur_len > 2:
                if i < len(tokenized_str) - 1 and \
                        tokenized_str[i + 1] in ["'", "▁'"]:
                    cur.append(tokenized_str[i + 1])
                    i += 1
                segments.append(cur)
                cur = []
                cur_len = 0
            continue
        else:
            sub = []
            for j in range(0, len(cur), max_seg):
                if j + max_seg < len(cur):
                    sub.append(cur[j:j + max_seg])
                else:
                    sub.append(cur[j:])
        segments.extend(sub)
        cur = []
        cur_len = 0
    if cur_len > 0:
        assert cur_len <= max_seg
        segments.append(cur)

    merged = []
    total_token = 0
    flags = []
    for seg in segments:
        total_token += len(seg)
        if len(seg) == 0:
            continue
        if len(merged) == 0:
            merged.append(seg)
            flags.append("first")
        elif (len(merged[-1]) + len(seg) <= max_seg
              and total_token > quick):
            merged[-1] = merged[-1] + seg
            flags.append("merge_quick")
        elif len(merged[-1]) + len(seg) <= max_seg / 2:
            merged[-1] = merged[-1] + seg
            flags.append("merge_half")
        else:
            merged.append(seg)
            flags.append("new")
    return merged, segments, total_token, flags


def _remove_long_silence(codes, silent, stop, maxc):
    code_lens = []
    codes_list = []
    counts = []
    kept = []
    isfix = False
    for code in codes:
        len_ = code.index(stop) if stop in code else len(code)
        count = sum(1 for v in code if v == silent)
        counts.append(count)
        if count > maxc:
            idxs = []
            n = 0
            for k in range(len_):
                assert code[k] != stop
                if code[k] != silent:
                    idxs.append(k)
                    n = 0
                elif n < 10:
                    idxs.append(k)
                    n += 1
            len_ = len(idxs)
            codes_list.append([code[k] for k in idxs])
            kept.append(idxs)
            isfix = True
        else:
            codes_list.append(code[:len_])
            kept.append(list(range(len_)))
        code_lens.append(len_)

    T = len(codes[0]) if codes else 0
    if isfix:
        L = max(len(c) for c in codes_list)
        out = [c + [stop] * (L - len(c)) for c in codes_list]
        rebuilt = True
    else:
        out = [list(c) for c in codes]
        rebuilt = False
    max_len = max(code_lens) if code_lens else 0
    if max_len < T:
        out = [c[:max_len] for c in out]
    return {"codes": out, "code_lens": code_lens, "isfix": isfix,
            "counts": counts, "kept_indices": kept, "rebuilt": rebuilt,
            "max_len": max_len}


def _mel_length_ledger(codes, stop, ratio, max_mel):
    has_warned = any(len(c) > 0 and c[-1] != stop for c in codes) \
        if codes else False
    code_lens = [c.index(stop) if stop in c else len(c) for c in codes]
    max_code_len = max(code_lens) if code_lens else 0
    truncated = [c[:max_code_len] for c in codes]
    target = [int(cl * ratio) for cl in code_lens]
    return {"code_lens": code_lens, "max_code_len": max_code_len,
            "truncated_codes": truncated, "has_warned": has_warned,
            "target_lengths": target, "total_frames": sum(target),
            "ratio": ratio, "max_mel_tokens": max_mel}


def solve(mode: str, **kw):
    if mode == "split_segments":
        tokenized = list(kw["tokenized"])
        max_seg = int(kw.get("max_text_tokens_per_segment", DEFAULT_MAX_SEG))
        split_tokens = kw.get("split_tokens")
        split_tokens = (list(PUNCTUATION_MARKS_TOKENS)
                        if split_tokens is None else list(split_tokens))
        quick = int(kw.get("quick_streaming_tokens", 0))
        merged, pre, total, flags = _split(tokenized, split_tokens,
                                           max_seg, quick)
        return {"segments": merged, "pre_merge": pre, "total_token": total,
                "n_segments": len(merged), "merge_flags": flags}

    if mode == "remove_long_silence":
        codes = [list(c) for c in kw["codes"]]
        silent = int(kw.get("silent_token", DEFAULT_SILENT_TOKEN))
        stop = kw["stop_mel_token"]
        maxc = int(kw.get("max_consecutive", DEFAULT_MAX_CONSECUTIVE))
        return _remove_long_silence(codes, silent, stop, maxc)

    if mode == "mel_length_ledger":
        codes = [list(c) for c in kw["codes"]]
        stop = kw["stop_mel_token"]
        ratio = kw.get("ratio", DEFAULT_RATIO)
        max_mel = int(kw.get("max_mel_tokens", DEFAULT_MAX_MEL_TOKENS))
        return _mel_length_ledger(codes, stop, ratio, max_mel)

    raise ValueError("unknown mode: %s" % mode)
