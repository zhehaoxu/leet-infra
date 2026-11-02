# FunASR FSMN-VAD Endpoint Detection Decision-Core Ledger

## Problem

FunASR's FSMN-VAD is the industry-standard streaming/offline endpoint detector: an FSMN network emits per-frame silence pdf probabilities followed by a three-layer pure-rule decision core — frame classification (probability gate + SNR gate + noise-dB EMA), a WindowDetector sliding-window state machine, and a top-level segment FSM. This problem hand-writes the full decision core (the waveform-buffer management layer is not modeled); no model, no real audio — decibels/scores are explicit inputs.

## Solution

Official semantics (modelscope/FunASR `funasr/models/fsmn_vad_streaming/model.py`, pinned line by line):

- **Frame classification**: `exp(log(1-score)) >= exp(log(score))·speech_2_noise_ratio + speech_noise_thres` passes the probability gate, then the SNR gate (`decibel - noise_avg >= snr_thres`); **only the probability-gate-failure branch updates the noise EMA** (SNR-gate and decibel-gate failures do not): `avg = (cur + avg·(N-1))/N`, with a direct assign on the first frame when `avg < -99.9`;
- **WindowDetector**: ring buffer + incremental win_sum; `pre=Sil and win_sum >= sil2sp_th → Sil2Speech`, `pre=Speech and win_sum <= sp2sil_th → Speech2Sil` — with equal thresholds (150/10=15) a win_sum exactly at the threshold flips with pre_state; **no forgetting while the ring is not full**;
- **Top FSM**: start = `max(0, cur - latency)` (latency = window + lookback/frm when do_extend); segments exceeding `max_single_segment_time/frm` force-cut (**only the Speech2Speech branch sets max_time_out**); Sil2Sil silence count reaching `max_end_sil_frame_cnt_thresh` closes with end = `cur - (thresh/frm - lookahead/frm - 1)`; **thresh = max_end_silence_time − speech_to_sil_time_thres** (the init_cache subtraction quirk, 800−150=650); tail extension when do_extend, not final, and silence ≤ lookahead frames;
- **NotDetected silence**: single-mode silence timeout (`> max_start_silence_time`) or final-frame fallback → **fake segment** (no real segment); multiple-mode ResetDetection on segment end — **does not clear** frm_cnt / noise EMA / number_ends; single mode skips frames after the terminal state;
- **Enum values**: change_states use the official AudioChangeState values (0=speech2speech 1=speech2sil 2=sil2sil 3=sil2speech 5=invalid).

## Data Scale

- Small-parameter anchor (win=5 frames, threshold 3, latency=10, end_lookback=4) hand-computed through the whole chain; official-default constant account (latency=40, thresh=650, end_lookback=54).
- Boundaries: SNR-gate failure not updating EMA, decibel-gate non-update, ring-buffer forgetting timeline, pure-silence fake segment, single-mode timeout, max_single force-cut, cross-segment EMA persistence (distinct noise dB per segment to make it observable).
- 200 randomized cross-checks (oracle transcribes the model.py decision layer verbatim).

## Scoring

All nine ledger keys match the official decision core: frame states / window outputs / segment boundaries (ms) exactly, noise_avg_db at 1e-9.

## Constraints

- No funasr / model forward calls; pure Python scalar accounting;
- scores domain (0,1) (0/1 would crash the official math.log; out of domain).
