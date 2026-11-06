"""Qwen3-ForcedAligner NAR 时间戳账（回源：QwenLM/Qwen3-ASR qwen3_forced_aligner.py）。

官方 Qwen3-ForcedAligner 是 LALM 式**非自回归**强制对齐器：把转写文本按词
（CJK 逐字、拉丁词）切开，词间与词尾各插两个 `<timestamp>` 占位；LLM 一次性
前向的 argmax 只在占位位置取值，乘 timestamp_segment_time 得到时间戳序列；
再经 LIS 单调修复与成对映射，输出每个词的 (start, end) 秒。本题重建其中
可判定的账：

1. **分词**（tokenize）：`tokenize_space_lang`——空格分段 → `clean_token`
   （仅保留 Unicode 字母 L* / 数字 N* / 撇号 `'`）→ `split_segment_with_chinese`
   （CJK 逐字切、非 CJK 连续段合并），空段丢弃。日/韩分支依赖 nagisa/
   soynlp 外部库，不进判题域（Chinese/English 均走 space-lang）。
2. **占位编码**（encode）：`"<timestamp><timestamp>".join(word_list)`
   + 尾部 `"<timestamp><timestamp>"`，前缀
   `<|audio_start|><|audio_pad|><|audio_end|>`——**每词恰 2 个时间戳位**。
3. **NAR 取值**（masked_timestamps）：`output_ids[input_ids == timestamp_token_id]`
   × segment_time（先掩码再乘，顺序不可换）。
4. **LIS 单调修复**（fix）：最长**非降**子序列 DP（`data[j] <= data[i]`、
   `dp[j]+1 > dp[i]` 才更新 parent，平分留先）、`dp.index(max(dp))` 取最早、
   parent 链回溯标记正常点；连续异常段按长度分治：
   - len<=2：左右最近正常值按 `(k-(i-1)) <= (j-k)` 距离择一（平分取左），
     单侧缺失取存在侧；
   - len>2：`step = (right-left)/(anomaly_count+1)` 线性插值，单侧平铺；
   - 最后 `[int(v) for v in result]` 截断（正浮点向零 = 向下取整）。
   左右值从 **result**（已修复值）而非原始数据读取。
5. **词 span 映射**（parse）：修复后每词取 `[2i]` / `[2i+1]`，
   `round(v / 1000.0, 3)` 转秒（银行家舍入）。

所有张量显式入参，fp64；字符串账与整数账为主，无 RNG（LLM 前向不入题，
argmax 结果作显式入参）。

判题约定：单入口 solve(mode, **kw)；CHECKS 列出全部校验函数。
"""

import unicodedata

import torch
from torch import Tensor

CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


TS = "<timestamp>"
AUDIO_PREFIX = "<|audio_start|><|audio_pad|><|audio_end|>"


def _is_cjk(ch: str) -> bool:
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF
        or 0x3400 <= code <= 0x4DBF
        or 0x20000 <= code <= 0x2A6DF
        or 0x2A700 <= code <= 0x2B73F
        or 0x2B740 <= code <= 0x2B81F
        or 0x2B820 <= code <= 0x2CEAF
        or 0xF900 <= code <= 0xFAFF
    )


def _is_kept(ch: str) -> bool:
    if ch == "'":
        return True
    cat = unicodedata.category(ch)
    return cat.startswith("L") or cat.startswith("N")


def _clean_token(token: str) -> str:
    return "".join(ch for ch in token if _is_kept(ch))


def _split_segment(seg: str) -> list:
    tokens = []
    buf = []
    for ch in seg:
        if _is_cjk(ch):
            if buf:
                tokens.append("".join(buf))
                buf = []
            tokens.append(ch)
        else:
            buf.append(ch)
    if buf:
        tokens.append("".join(buf))
    return tokens


def solve(mode: str, **kw):
    if mode == "tokenize":
        text = kw["text"]
        word_list = []
        for seg in text.split():
            cleaned = _clean_token(seg)
            if cleaned:
                word_list.extend(_split_segment(cleaned))
        return {"word_list": word_list}

    if mode == "encode":
        word_list = kw["word_list"]
        body = (TS + TS).join(word_list) + TS + TS
        return {"text": AUDIO_PREFIX + body}

    if mode == "masked_timestamps":
        input_ids = kw["input_ids"]          # (L,)
        output_ids = kw["output_ids"]        # (L,) argmax 结果
        tid = kw["timestamp_token_id"]
        segment_time = kw["segment_time"]
        masked = output_ids[input_ids == tid]
        return {"values": masked.to(torch.float64) * segment_time,
                "count": int(masked.numel())}

    if mode == "fix":
        data = kw["timestamp"].tolist()
        n = len(data)
        dp = [1] * n
        parent = [-1] * n
        for i in range(1, n):
            for j in range(i):
                # 非降 LIS：允许相等；严格大于才更新 parent（平分留先）
                if data[j] <= data[i] and dp[j] + 1 > dp[i]:
                    dp[i] = dp[j] + 1
                    parent[i] = j
        max_length = max(dp)
        max_idx = dp.index(max_length)       # 平分取最早
        lis_indices = []
        idx = max_idx
        while idx != -1:
            lis_indices.append(idx)
            idx = parent[idx]
        lis_indices.reverse()
        is_normal = [False] * n
        for idx in lis_indices:
            is_normal[idx] = True
        result = data.copy()
        i = 0
        while i < n:
            if not is_normal[i]:
                j = i
                while j < n and not is_normal[j]:
                    j += 1
                anomaly_count = j - i
                if anomaly_count <= 2:
                    left_val = None
                    for k in range(i - 1, -1, -1):
                        if is_normal[k]:
                            left_val = result[k]
                            break
                    right_val = None
                    for k in range(j, n):
                        if is_normal[k]:
                            right_val = result[k]
                            break
                    for k in range(i, j):
                        if left_val is None:
                            result[k] = right_val
                        elif right_val is None:
                            result[k] = left_val
                        else:
                            # 距左正常点 (k-(i-1)) <= 距右正常点 (j-k) 时取左
                            result[k] = left_val if (k - (i - 1)) <= (j - k) else right_val
                else:
                    left_val = None
                    for k in range(i - 1, -1, -1):
                        if is_normal[k]:
                            left_val = result[k]
                            break
                    right_val = None
                    for k in range(j, n):
                        if is_normal[k]:
                            right_val = result[k]
                            break
                    if left_val is not None and right_val is not None:
                        step = (right_val - left_val) / (anomaly_count + 1)
                        for k in range(i, j):
                            result[k] = left_val + step * (k - i + 1)
                    elif left_val is not None:
                        for k in range(i, j):
                            result[k] = left_val
                    elif right_val is not None:
                        for k in range(i, j):
                            result[k] = right_val
                i = j
            else:
                i += 1
        return {"fixed": [int(v) for v in result], "dp": dp,
                "parent": parent, "is_normal": is_normal,
                "lis_indices": lis_indices}

    if mode == "parse":
        word_list = kw["word_list"]
        fixed = solve("fix", timestamp=kw["timestamp"])["fixed"]
        spans = []
        for i, word in enumerate(word_list):
            spans.append({
                "text": word,
                "start_time": round(fixed[i * 2] / 1000.0, 3),
                "end_time": round(fixed[i * 2 + 1] / 1000.0, 3),
            })
        return {"spans": spans, "fixed": fixed}

    raise ValueError(f"unknown mode: {mode}")
