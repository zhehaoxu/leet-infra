"""Moonshine streaming verify test cases — judge-style: check(solve) + string CHECKS.
Semantics pinned by core/moonshine-streaming-model.cpp (encode / decode_full)."""


def check_stable_frames(solve):
    # empty audio emits nothing
    assert solve(mode="stable_frames", total=0, emitted=0, lookahead=16, is_final=False) == 0
    # lookahead holds back the last 16 frames until final
    assert solve(mode="stable_frames", total=30, emitted=0, lookahead=16, is_final=False) == 14
    assert solve(mode="stable_frames", total=46, emitted=14, lookahead=16, is_final=False) == 16
    # not enough audio yet: lookahead swallows everything
    assert solve(mode="stable_frames", total=10, emitted=0, lookahead=16, is_final=False) == 0
    # nothing new since the last commit
    assert solve(mode="stable_frames", total=20, emitted=14, lookahead=16, is_final=False) == 0
    # is_final commits everything accumulated
    assert solve(mode="stable_frames", total=30, emitted=14, lookahead=16, is_final=True) == 16
    assert solve(mode="stable_frames", total=10, emitted=0, lookahead=16, is_final=True) == 10


def check_encoder_window(solve):
    # left context = 16 frames per layer; start clamps at 0
    out = solve(mode="encoder_window", emitted=0, total=40, depth=6)
    assert out == {"window_start": 0, "window_size": 40}, out
    out = solve(mode="encoder_window", emitted=60, total=100, depth=3)
    assert out == {"window_start": 12, "window_size": 88}, out  # 60 - 48
    out = solve(mode="encoder_window", emitted=30, total=50, depth=1)
    assert out == {"window_start": 14, "window_size": 36}, out  # 30 - 16
    out = solve(mode="encoder_window", emitted=90, total=100, depth=6)
    assert out == {"window_start": 0, "window_size": 100}, out  # 90 - 96 < 0


def check_max_tokens(solve):
    # 20 ms per frame, 6.5 tokens per second: 100 frames -> 2 s -> 13 tokens
    assert solve(mode="max_tokens", memory_len=100) == 13
    assert solve(mode="max_tokens", memory_len=1000) == 130
    # ceil, not floor: 101 frames -> 13.13 s -> 14
    assert solve(mode="max_tokens", memory_len=101) == 14
    # capped at max_seq_len
    assert solve(mode="max_tokens", memory_len=10000) == 448
    # explicit rate/frame overrides
    assert solve(mode="max_tokens", memory_len=100, frame_ms=10.0, rate=10.0) == 10


def check_cache_policy(solve):
    # memory append invalidates; compute rebuilds; decoder_reset keeps cross K/V
    ops = [("encode", 5), "compute", "decoder_reset", ("encode", 3), ("encode", 0), "compute"]
    assert solve(mode="cache_policy", ops=ops) == [False, True, True, False, False, True]
    # a frame-less encode leaves the valid flag untouched
    ops2 = ["compute", ("encode", 0), "decoder_reset"]
    assert solve(mode="cache_policy", ops=ops2) == [True, True, True]
    # unknown op raises
    try:
        solve(mode="cache_policy", ops=["transcribe"])
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def check_diverge_point(solve):
    # first mismatch stops the count
    assert solve(mode="diverge_point", draft=[5, 6, 7], predictions=[5, 6, 9]) == 2
    assert solve(mode="diverge_point", draft=[5, 6, 7], predictions=[9, 6, 7]) == 0
    assert solve(mode="diverge_point", draft=[5, 6], predictions=[5, 6]) == 2
    assert solve(mode="diverge_point", draft=[], predictions=[]) == 0


def check_verify_decode(solve):
    bos, eos, vocab = 1, 2, 8

    # A: no biaser, full match -> continue from the final prediction
    out = solve(mode="verify_decode", draft=[3, 4],
                verify_logits=[[0, 0, 0, 5, 0, 0, 0, 0],
                               [0, 0, 0, 0, 5, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 5, 0]],
                ar_logits=[[0, 0, 0, 0, 0, 0, 0, 5],
                           [0, 0, 5, 0, 0, 0, 0, 0]],
                bos=bos, eos=eos, vocab_size=vocab, max_tokens=16)
    assert out == {"tokens": [6, 7], "accepted": [3, 4], "cache_reset": False}, out

    # B: divergence -> self-KV reset, re-run accepted prefix, continue from rerun row
    out = solve(mode="verify_decode", draft=[3, 4],
                verify_logits=[[0, 0, 0, 5, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 5],
                               [0, 0, 0, 0, 0, 0, 0, 0]],
                rerun_logits=[[0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 5, 0, 0]],
                ar_logits=[[0, 0, 5, 0, 0, 0, 0, 0]],
                bos=bos, eos=eos, vocab_size=vocab, max_tokens=16)
    assert out == {"tokens": [5], "accepted": [3], "cache_reset": True}, out
    # (the AR row after 5 argues eos -> the loop stops without appending eos)
    out_b2 = solve(mode="verify_decode", draft=[3, 4],
                   verify_logits=[[0, 0, 0, 5, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 5],
                                  [0, 0, 0, 0, 0, 0, 0, 0]],
                   rerun_logits=[[0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 5, 0, 0]],
                   ar_logits=[[0, 0, 1, 0, 0, 0, 0, 0]],
                   bos=bos, eos=eos, vocab_size=vocab, max_tokens=16)
    assert out_b2["tokens"] == [5], out_b2

    # C: the biaser flips verification to accept (bonus beats a stronger rival)
    biaser = {"children": [{3: 1}, {4: 2}, {}], "depths": [0, 1, 2], "boost": 2.0, "active": [0]}
    out = solve(mode="verify_decode", draft=[3, 4],
                verify_logits=[[0, 0, 0, 1, 0, 0, 0, 0],
                               [0, 0, 0, 0, 1, 0, 2.5, 0],
                               [0, 0, 0, 0.5, 0, 0, 0, 1]],
                ar_logits=[[0, 0, 0, 0, 0.5, 0, 0.5, 0],
                           [0, 0, 5, 0.5, 0, 0, 0, 0]],
                bos=bos, eos=eos, vocab_size=vocab, max_tokens=16, biaser=biaser)
    # row1: 4 gets +2*(1+ln2)=3.386 -> 4.386 beats 2.5 -> full match;
    # row2: 3 gets +2.0 -> 2.5 beats 1.0 -> continue from 3;
    # ar0: 4 gets +3.386 -> 3.886 beats 0.5 -> push 4;
    # ar1: eos=5.0 beats biased 3 (0.5+2.0) -> stop
    assert out["accepted"] == [3, 4] and out["cache_reset"] is False, out
    assert out["tokens"] == [3, 4], out

    # D: max_tokens truncates the AR continuation even with logits left
    out = solve(mode="verify_decode", draft=[3],
                verify_logits=[[0, 0, 0, 5, 0, 0, 0, 0], [0, 0, 0, 0, 0, 5, 0, 0]],
                ar_logits=[[0, 0, 0, 0, 0, 0, 0, 5]] * 5,
                bos=bos, eos=eos, vocab_size=vocab, max_tokens=3)
    assert out["tokens"] == [5, 7, 7] and out["cache_reset"] is False, out

    # E: empty draft behaves like regular decoding from BOS
    out = solve(mode="verify_decode", draft=[],
                verify_logits=[[0, 0, 0, 0, 0, 5, 0, 0]],
                ar_logits=[[0, 0, 5, 0, 0, 0, 0, 0]],
                bos=bos, eos=eos, vocab_size=vocab, max_tokens=16)
    assert out["tokens"] == [5] and out["accepted"] == [] and out["cache_reset"] is False, out

    # F: an EOS prediction as the continue point emits nothing
    out = solve(mode="verify_decode", draft=[3],
                verify_logits=[[0, 0, 0, 5, 0, 0, 0, 0], [0, 0, 5, 0, 0, 0, 0, 0]],
                ar_logits=[],
                bos=bos, eos=eos, vocab_size=vocab, max_tokens=16)
    assert out["tokens"] == [] and out["accepted"] == [3], out

    # B3: divergence with a biaser — the walk is REWOUND to the accepted prefix
    # (without the rewind the bonus lands on the wrong token)
    biaser_b = {"children": [{3: 1}, {4: 2}, {}], "depths": [0, 1, 2], "boost": 2.0, "active": [0]}
    out = solve(mode="verify_decode", draft=[3, 4],
                verify_logits=[[0, 0, 0, 5, 0, 0, 0, 0],
                               [0, 0, 0, 0, 1, 0, 0, 5],
                               [0, 0, 0, 0, 0, 0, 0, 0]],
                rerun_logits=[[0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 2, 0]],
                ar_logits=[[0, 0, 5, 0, 0, 0, 0, 0]],
                bos=bos, eos=eos, vocab_size=vocab, max_tokens=16, biaser=biaser_b)
    # row1: 9 (5.0) beats biased 4 (1+3.386=4.386) -> diverge at 1; rewind
    # re-walks [3] so rerun row1: 4 -> 1+3.386=4.386 beats 7 -> continue from 4
    assert out["accepted"] == [3] and out["cache_reset"] is True, out
    assert out["tokens"] == [4], out

    # G: a token proposed by two active paths takes the LARGER bonus (never stacked)
    biaser_g = {"children": [{3: 1, 6: 3}, {4: 2}, {}, {3: 4}, {}],
                "depths": [0, 1, 2, 1, 2], "boost": 2.0, "active": [0]}
    out = solve(mode="verify_decode", draft=[6, 3],
                verify_logits=[[0, 0, 0, 0.5, 0, 0, 1, 0],
                               [0, 0, 0, 0.5, 0, 0, 0, 4],
                               [0, 0, 0, 0, 0.5, 1, 0, 0]],
                rerun_logits=[[0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 5, 0, 0, 4, 0]],
                ar_logits=[[0, 0, 6, 0, 0.5, 0, 0, 0]],
                bos=bos, eos=eos, vocab_size=vocab, max_tokens=16, biaser=biaser_g)
    # row1: 3 = 0.5 + max(2 [root], 3.386 [inside 6->3]) = 3.886 < 4.0 (7 wins)
    # -> diverge at 1; stacked it would be 5.886 and win. Rewind walks [6];
    # rerun row1: 3 = 5+3.386 -> continue from 3; ar row: eos 6.0 beats 3.886.
    assert out["accepted"] == [6] and out["cache_reset"] is True, out
    assert out["tokens"] == [3], out


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
