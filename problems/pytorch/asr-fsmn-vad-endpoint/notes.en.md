# notes — asr-fsmn-vad-endpoint (order 716)

## Sourcing

- Authoritative source: modelscope/FunASR `funasr/models/fsmn_vad_streaming/model.py` (main, 2026, 1302 lines). Decision core trio: `GetFrameState` / `WindowDetector` / `DetectOneFrame` (FSM).
- **nn_eval_block_size overwritten per chunk**: `ComputeScores` sets `self.vad_opts.nn_eval_block_size = scores.shape[1]` — DetectCommonFrames processes the whole new chunk per call (the reverse range over frm_cnt-1-i is actually ascending). Offline single-pass ≡ per-frame ascending with is_final only on the last frame.
- **Threshold subtraction quirk**: `max_end_sil_frame_cnt_thresh = max_end_silence_time - speech_to_sil_time_thres` (init_cache, 800−150=650) — any account using only max_end_silence_time=800 gets the closing timing wrong.
- **End lookback**: lookback = thresh/frm − lookahead/frm − 1 (do_extend); official default 65−10−1=54 frames.
- **Narrow EMA update surface**: only the probability-gate-failure else branch updates; SNR-gate / decibel-gate failures return Sil with **EMA untouched**. The EMA persists across segments (ResetDetection doesn't clear it).
- **max_time_out set only in the Speech2Speech branch** — the Sil2Speech/Speech2Sil/Sil2Sil force-cut branches don't set it (official asymmetry).
- **Fake segment**: pure-silence final-frame fallback runs OnVoiceStart(0,fake)+OnVoiceEnd(0,fake); fake produces no real segment but increments number_ends and enters the terminal state; single mode additionally has the silence-timeout fake.
- **Dynamic silence schedule** (inference layer): when the user doesn't pass max_end_silence_time, offline mode enables DEFAULT_SILENCE_SCHEDULE and forces speech_noise_thres=0.5, accumulated per chunk (default 60s!) — deeply coupled with frame decisions; **not modeled here** (passing max_end_silence_time explicitly takes the fixed-threshold path); noted for reference.
- **Buffer-layer stripping**: PopDataBufTillFrame/DropCachedFrames only affect waveform slicing and data_buf_start_frame (the first argument of start = max(0, cur-latency) stays 0), never segment boundaries — decision-core equivalent.

## Test Design

- Oracle transcribes the model.py decision layer verbatim (enums and branch order identical); a small-parameter anchor is hand-computed (win=5, threshold 3 — the ring buffer doesn't forget while unfilled, which the first draft's hand calc missed).
- SNR-gate case: db/score crafted so the probability gate passes but the SNR gate fails (after EMA direct-assign, db=-31, snr_thres=0), asserting EMA unchanged — the hook for the ema-update-on-snr-gate mutation.
- Cross-segment EMA case: distinct noise dB per segment (−30/−40), asserting the final value ∈ (−40, −30) — if the EMA were reset, the first silence frame would direct-assign −40, immediately distinguishable.

## Mutations (12 caught + 1 equivalent)

- Catching hinges on: the EMA update surface (three gate branches × **full formula** — the first draft's mutations only wrote the direct-assign branch and were accidentally equivalent), window threshold polarities, ring forgetting, start latency, end lookback, threshold subtraction, tail-extension off, single-mode skip, fake→real, EMA first-frame direct assign, cross-segment EMA reset (**the mutation must include both the nonlocal declaration and the assignment — either alone is a no-op/local-variable; two false SURVIVEDs**).
- 1 equivalent mutation (SURVIVED, recorded): tail-extension off — extension only writes latest_speech_frame, and OnVoiceEnd's backfill loop (latest+1..end−1) plus end=cur−lookback already covers the same frames; decision-core segment boundaries unchanged (in the full pipeline it affects output buffer contents, so it is not equivalent for the official code).

## Pitfalls

- Nested functions mutating the outer `state` need nonlocal (the first draft missed it in detect_one_frame → UnboundLocalError).
- Hand-computing the ring window: an unfilled window forgets nothing (at frame 4 the old speech frames are still inside, win_sum doesn't drop).
- Boundary with 693/694 (specaugment/cif): this problem covers the VAD endpoint state machine; together with 670/700/714/715 it completes the Whisper/FunASR dual-track asr ledger.
