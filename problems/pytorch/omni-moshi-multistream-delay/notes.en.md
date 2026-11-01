# Moshi multistream delay ledger (notes)

Source: kyutai-labs/moshi `rust/moshi-core/src/lm_generate_multistream.rs`
(Config::v0_1: 8 generated + 8 input codebooks, vocab 2049, audio_pad = vocab-1 = 2048,
acoustic_delay 2, text_pad 3, text_eop 0, text_start 32000).

- **Write-side delay asymmetry**: cb0 writes with zero delay (real-time stream), cb1-7
  lag by 2; the input half cb8-15 writes every frame directly (no delay). The read side
  (model input codes) additionally reads cb0/cb8 with zero delay and the rest from
  step-delay-1 (out of scope here; stripped into inputs).
- **Overwrite chain**: saturating_sub makes cb1-7 of steps 0..2 all write slot 0 — the
  official comment reads "Overwrite existing positions even if there are
  non-UNGENERATED values". Anchor: slot0 = [step0.cb0, step2.cb1-7].
- **Triple playback gate**: None before step <= delay; read frame step-delay-1; any
  audio_pad in the frame (from dep sampling 2048 directly or the None fallback) mutes
  the whole frame. The any->all and >=-> threshold mutations are both anchored.
- **UNGENERATED**: the official matrix starts at u32::MAX and the read side bails on
  it (inputs here guarantee the needed cells are covered; leftovers appear only in
  tail rows and short input streams, and -1 < 2048 does not mute — the convention is
  written into the problem statement).
- Equivalent mutation: `list(frame)`->`frame` (no later write hits a recorded slot).
  The text stream (text_pad 3's pad_mult sampling bias, repetition penalty's non-pad
  skip) belongs to the sampling side and is out of scope.
