import math

import torch


def solve(emission, tokens, transcript, separator="|"):
    """WhisperX 强制对齐账（m-bain/whisperX whisperx/alignment.py，
    get_trellis / backtrack / merge_repeats / merge_words 的 torch
    fallback 口径）。emission 为 (T, V) log 概率张量（log-softmax），
    tokens 为词表下标列表，transcript 为等长标签列表，separator 为
    词分隔标签（默认 "|"）。

    官方口径（v1.5.x alignment.py 钉死）：

      get_trellis：trellis 形状 (T+1, N+1)（时间与词表各多一维：
        时间维简化代码，词表维代表 <SoS>）
        trellis[0,0] = 0；trellis[1:, 0] = cumsum(emission[:, blank])
        trellis[0, -N:] = -inf（第 0 行词表列）
        trellis[-N:, 0] = **+inf**（列 0 尾部 N 行置 +inf——强制
          收尾机制：j==1 回溯到尾行时 changed=+inf 必 > stayed，
          逼路径在末尾收束；不设则可能走不完）
        递推：trellis[t+1, j] = max(stay, change)
          stay   = trellis[t, j]   + emission[t, blank]
          change = trellis[t, j-1] + emission[t, tokens[j-1]]
          （平分取 max 留在原语序——max(stay, change) 首参 stay）
      backtrack：j = 最后一列；t_start = argmax(trellis[:, j])
        （并列取**首个**最大值行）；从 t_start 向下走：
        stayed  = trellis[t-1, j]   + emission[t-1, blank]
        changed = trellis[t-1, j-1] + emission[t-1, tokens[j-1]]
        lbl = tokens[j-1] if changed > stayed else blank
          （**严格 >**：平分留 blank/stay，不前进）
        prob = exp(emission[t-1, lbl])（帧概率是概率域，trellis 是
          log 域——两域混用是官方原样）
        点 (token_index=j-1, time_index=t-1)；changed > stayed 时
        j -= 1，j 归 0 -> ok；循环走完未归 0 -> **None**（T < N 时
        列全 -inf、argmax 取 0 行、空路径 -> None）
      merge_repeats：按 token_index 连续段合并；
        segment = [首帧 time, 末帧 time + 1)（左闭右开）
        score = 段内帧 prob 的**算术平均**（不按长度加权）
      merge_words：按 separator 切分；word 标签 = 字符段标签拼接；
        span = [首段 start, 末段 end)；score = **长度加权**平均
        (sum(seg.score * seg.length) / sum(seg.length))——与
        merge_repeats 的均值是两套聚合
        首元素即 separator / 连续 separator -> i1 == i2，不产空词；
        尾 separator 后无残段 -> 不产词

    需要产出的账（键固定）：
      ok               回溯是否收束（bool）
      path             [[token_index, time_index, prob], ...]（时间升序；
                       not ok 时 []）
      trellis_corner   trellis[T, N]（float；T >= N 时为有限值）
      col0_inf         列 0 中 +inf 的条数（= N，尾部强制收尾填充）
      segments         [[label, start, end, score], ...]（not ok 时 []）
      words            [[label, start, end, score], ...]（not ok 时 []）

    红线：不得调用 torchaudio / torio / numba / whisperx 的任何函数；
    仅 torch 张量运算 + 纯 Python 循环。
    """
    if not isinstance(emission, torch.Tensor):
        raise ValueError("emission must be a torch.Tensor")
    if not isinstance(separator, str):
        raise ValueError("separator must be a str")
    T = int(emission.shape[0])
    V = int(emission.shape[1])
    N = len(tokens)
    if N == 0 or len(transcript) != N:
        raise ValueError("tokens/transcript must be non-empty and equal length")
    blank_id = 0

    # ---- get_trellis ----
    trellis = torch.empty((T + 1, N + 1), dtype=torch.float64)
    trellis[0, 0] = 0.0
    trellis[1:, 0] = torch.cumsum(emission[:, blank_id].double(), 0)
    trellis[0, -N:] = -float("inf")
    trellis[-N:, 0] = float("inf")
    for t in range(T):
        for j in range(1, N + 1):
            stay = float(trellis[t, j]) + float(emission[t, blank_id])
            change = float(trellis[t, j - 1]) + float(emission[t, tokens[j - 1]])
            trellis[t + 1, j] = stay if stay > change else change

    # ---- backtrack ----
    j = N + 1 - 1
    col = trellis[:, j]
    t_start = int(torch.argmax(col).item())
    path = []
    ok = False
    t = t_start
    while t > 0:
        stayed = float(trellis[t - 1, j]) + float(emission[t - 1, blank_id])
        changed = float(trellis[t - 1, j - 1]) + float(emission[t - 1, tokens[j - 1]])
        if changed > stayed:
            lbl = tokens[j - 1]
        else:
            lbl = blank_id
        prob = math.exp(float(emission[t - 1, lbl]))
        path.append((j - 1, t - 1, prob))
        if changed > stayed:
            j -= 1
            if j == 0:
                ok = True
                break
        t -= 1
    path.reverse()

    # ---- merge_repeats ----
    segments = []
    if ok:
        i1, i2 = 0, 0
        while i1 < len(path):
            while i2 < len(path) and path[i1][0] == path[i2][0]:
                i2 += 1
            score = sum(p[2] for p in path[i1:i2]) / (i2 - i1)
            segments.append((transcript[path[i1][0]], path[i1][1],
                             path[i2 - 1][1] + 1, score))
            i1 = i2

    # ---- merge_words ----
    words = []
    if ok:
        i1, i2 = 0, 0
        while i1 < len(segments):
            if i2 >= len(segments) or segments[i2][0] == separator:
                if i1 != i2:
                    segs = segments[i1:i2]
                    word = "".join(s[0] for s in segs)
                    denom = sum(s[2] - s[1] for s in segs)
                    score = sum(s[3] * (s[2] - s[1]) for s in segs) / denom
                    words.append((word, segs[0][1], segs[-1][2], score))
                i1 = i2 + 1
                i2 = i1
            else:
                i2 += 1

    col0 = trellis[:, 0]
    return {
        "ok": ok,
        "path": [[p[0], p[1], p[2]] for p in path],
        "trellis_corner": float(trellis[T, N]),
        "col0_inf": int(sum(1 for x in col0.tolist() if math.isinf(x) and x > 0)),
        "segments": [list(s) for s in segments],
        "words": [list(w) for w in words],
    }
