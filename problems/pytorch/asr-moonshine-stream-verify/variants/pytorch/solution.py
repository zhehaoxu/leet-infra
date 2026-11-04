"""Moonshine streaming transcribe — stable frames, cache lifecycle, greedy draft verify.

Reproduces the pure-function core of moonshine-ai/moonshine
core/moonshine-streaming-model.cpp (the streaming ASR state machine):

- stable_frames: the encoder only commits "stable" frames — is_final commits
  everything, otherwise max(0, total - lookahead) (total_lookahead = 16 frames
  of encoder lookahead). new_frames = stable - already emitted; <= 0 emits
  nothing. total == 0 emits nothing.
- encoder_window: the encoder re-runs on a sliding window with a fixed per-layer
  left context: left = 16 * depth; window_start = max(0, emitted - left);
  window_size = total - window_start.
- max_tokens: decode budget = ceil(memory_len * 0.020 * 6.5) capped at
  max_seq_len (448) — 20 ms per frame, 6.5 tokens per second of speech.
- cache_policy: cross-attention K/V are precomputed from memory. Appending to
  memory (encode with new_frames > 0) invalidates them; encode that produces no
  frames leaves the flag untouched; compute_cross_kv rebuilds and marks valid;
  decoder_reset clears only the self-attention KV cache and keeps cross K/V
  valid (memory unchanged).
- diverge_point: greedy speculative verification — the number of leading draft
  tokens matching the teacher-forced predictions (first mismatch stops).
- verify_decode: full greedy draft-verify-and-continue ledger. Verification
  runs the decoder on [BOS] + draft and takes per-row argmax. The context
  biaser walks the TEACHER-FORCED PREFIX (advance with tokens_with_bos[t+1]),
  not the predictions, because position t is conditioned on the prefix. Full
  match: continue auto-regressively from the final prediction. Divergence:
  reset the self-attention KV cache (cache_reset = True), rewind the biaser
  walk to the accepted prefix, re-run [BOS] + accepted and continue from the
  biased argmax at that row. The AR loop pushes each token (never EOS), walks
  the biaser, and consumes the next logits row in order until EOS or the
  max_tokens budget.

All logits are explicit inputs (rows over a small vocab); greedy argmax only —
the sampled-verification family lives in the speculative-decoding collection.
"""

import math


def _argmax(row):
    best = 0
    for i in range(1, len(row)):
        if row[i] > row[best]:
            best = i
    return best


def _biased_argmax(row, biaser, vocab_size):
    if biaser is not None:
        children, boost = biaser["children"], biaser["boost"]
        pending = {}
        for node_index in biaser["active"]:
            depth = biaser["depths"][node_index] + 1
            bonus = boost * (1.0 + math.log(depth)) if depth > 0 else 0.0
            for token in children[node_index]:
                if 0 <= token < vocab_size:
                    pending[token] = max(pending.get(token, bonus), bonus)
        row = list(row)
        for token, bonus in pending.items():
            row[token] += bonus
    return _argmax(row)


def _biaser_advance(biaser, token):
    if biaser is None:
        return
    next_active = [0]
    for node_index in biaser["active"]:
        child = biaser["children"][node_index].get(token)
        if child is not None:
            next_active.append(child)
    biaser["active"] = next_active


def solve(mode, **kw):
    if mode == "stable_frames":
        total, emitted = kw["total"], kw["emitted"]
        lookahead, is_final = kw["lookahead"], kw["is_final"]
        if total == 0:
            return 0
        stable = total if is_final else max(0, total - lookahead)
        new_frames = stable - emitted
        return new_frames if new_frames > 0 else 0

    if mode == "encoder_window":
        emitted, total, depth = kw["emitted"], kw["total"], kw["depth"]
        left = 16 * depth
        window_start = max(0, emitted - left)
        return {"window_start": window_start, "window_size": total - window_start}

    if mode == "max_tokens":
        memory_len = kw["memory_len"]
        frame_ms = kw.get("frame_ms", 20.0)
        rate = kw.get("rate", 6.5)
        cap = kw.get("cap", 448)
        duration = memory_len * frame_ms / 1000.0
        return min(int(math.ceil(duration * rate)), cap)

    if mode == "cache_policy":
        valid = False
        states = []
        for op in kw["ops"]:
            kind, produced = (op if isinstance(op, tuple) else (op, None))
            if kind == "encode":
                if (produced or 0) > 0:
                    valid = False  # memory changed
                # a frame-less encode touches nothing
            elif kind == "compute":
                valid = True
            elif kind == "decoder_reset":
                pass  # keeps cross K/V valid: memory unchanged
            else:
                raise ValueError(f"unknown op: {op}")
            states.append(valid)
        return states

    if mode == "diverge_point":
        draft, predictions = kw["draft"], kw["predictions"]
        point = 0
        for i in range(len(draft)):
            if predictions[i] == draft[i]:
                point = i + 1
            else:
                break
        return point

    if mode == "verify_decode":
        draft = kw["draft"]
        verify_logits = kw["verify_logits"]
        rerun_logits = kw.get("rerun_logits") or []
        ar_logits = kw.get("ar_logits") or []
        bos, eos = kw["bos"], kw["eos"]
        vocab_size = kw["vocab_size"]
        max_tokens = kw["max_tokens"]
        biaser = kw.get("biaser")
        if biaser is not None:
            biaser["active"] = [0]

        tokens_with_bos = [bos] + list(draft)
        predictions = []
        for t in range(len(tokens_with_bos)):
            predictions.append(_biased_argmax(verify_logits[t], biaser, vocab_size))
            if biaser is not None and t + 1 < len(tokens_with_bos):
                _biaser_advance(biaser, tokens_with_bos[t + 1])

        point = 0
        for i in range(len(draft)):
            if predictions[i] == draft[i]:
                point = i + 1
            else:
                break
        accepted = list(draft[:point])

        result = []
        cache_reset = False

        def continue_ar(start_token):
            current = start_token
            while current != eos and len(result) < max_tokens:
                result.append(current)
                _biaser_advance(biaser, current)
                if not ar_logits:
                    return
                current = _biased_argmax(ar_logits.pop(0), biaser, vocab_size)

        if point == len(draft):
            final_pred = predictions[len(draft)]
            continue_ar(final_pred)
        else:
            cache_reset = True
            if biaser is not None:
                biaser["active"] = [0]
                for tok in accepted:
                    _biaser_advance(biaser, tok)
            new_pred = _biased_argmax(rerun_logits[point], biaser, vocab_size)
            continue_ar(new_pred)

        return {"tokens": result, "accepted": accepted, "cache_reset": cache_reset}

    raise ValueError(f"unknown mode: {mode}")


CHECKS = [
    "check_stable_frames",
    "check_encoder_window",
    "check_max_tokens",
    "check_cache_policy",
    "check_diverge_point",
    "check_verify_decode",
]


def check(solve):
    for name in CHECKS:
        globals()[name](solve)
