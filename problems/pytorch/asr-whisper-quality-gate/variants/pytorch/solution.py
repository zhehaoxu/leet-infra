import zlib


def _cr(text):
    b = text.encode("utf-8")
    return len(b) / len(zlib.compress(b))


def solve(windows, temperatures=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
          compression_ratio_threshold=2.4, logprob_threshold=-1.0,
          no_speech_threshold=0.6):
    if not isinstance(temperatures, (list, tuple)) or len(temperatures) == 0:
        raise ValueError("temperatures must be a non-empty list")
    for t in temperatures:
        if isinstance(t, bool) or not isinstance(t, (int, float)) \
                or isinstance(t, float) and (t != t or t in (float("inf"), float("-inf"))):
            raise ValueError("temperatures entries must be finite numbers")
        if t < 0:
            raise ValueError("temperatures entries must be non-negative")
    for name, thr in (("compression_ratio_threshold", compression_ratio_threshold),
                      ("logprob_threshold", logprob_threshold),
                      ("no_speech_threshold", no_speech_threshold)):
        if isinstance(thr, bool) or not isinstance(thr, (int, float)) \
                or isinstance(thr, float) and (thr != thr or thr in (float("inf"), float("-inf"))):
            raise ValueError(f"{name} must be a finite number")
    if not isinstance(windows, (list, tuple)) or len(windows) == 0:
        raise ValueError("windows must be a non-empty list")

    n_t = len(temperatures)
    ledgers = []
    for w in windows:
        if not isinstance(w, dict) or set(w.keys()) != {"texts", "avg_logprob", "no_speech_prob"}:
            raise ValueError("window needs exactly keys texts/avg_logprob/no_speech_prob")
        texts, alps, nsps = w["texts"], w["avg_logprob"], w["no_speech_prob"]
        for seq, name in ((texts, "texts"), (alps, "avg_logprob"), (nsps, "no_speech_prob")):
            if not isinstance(seq, (list, tuple)) or len(seq) != n_t:
                raise ValueError(f"{name} must have one entry per temperature")
        for x in texts:
            if not isinstance(x, str):
                raise ValueError("texts entries must be str")
        for x in alps + list(nsps):
            if isinstance(x, bool) or not isinstance(x, (int, float)) \
                    or isinstance(x, float) and (x != x or x in (float("inf"), float("-inf"))):
                raise ValueError("metrics must be finite numbers")

        crs = [_cr(x) for x in texts]

        # decode_with_fallback
        chain = []
        selected = None
        silence_covered = False
        for i in range(n_t):
            chain.append(i)
            nf = crs[i] > compression_ratio_threshold or alps[i] < logprob_threshold
            covered = nsps[i] > no_speech_threshold and alps[i] < logprob_threshold
            if covered:
                nf = False  # silence overrides
            if not nf:
                selected = i
                silence_covered = covered
                break
        if selected is None:
            selected = n_t - 1
            silence_covered = (nsps[selected] > no_speech_threshold
                               and alps[selected] < logprob_threshold)

        # segment-level skip
        should_skip = nsps[selected] > no_speech_threshold
        if alps[selected] > logprob_threshold:
            should_skip = False

        ledgers.append({
            "compression_ratios": crs,
            "selected_temp_index": selected,
            "fallback_chain": chain,
            "silence_covered": silence_covered,
            "action": "skip" if should_skip else "transcribe",
        })
    return ledgers
