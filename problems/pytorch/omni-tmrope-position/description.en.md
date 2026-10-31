# TMRoPE: 3D Position IDs for Text + Audio Segments

## Background

Qwen2.5-Omni's thinker LLM uses **TMRoPE**: three-axis (t/h/w) position ids where vision tokens expand into a 3D grid and everything else degenerates to 1D with identical rows. HF transformers `modeling_qwen2_5_omni.py` `get_rope_index` is exactly this ledger: scan special tokens in input order, append per-segment position ids into `llm_pos_ids_list`, and stack into `[3, bs, seq]`.

This problem transcribes the **text + standalone-audio** path (`use_audio_in_video=False`, no image/video): audio is a first-class citizen — an `<|audio_bos|> <audio_pad|> <|audio_eos|>` three-token block whose pad is expanded into `audio_len` audio tokens during forward.

## Exact Ledger

1. **Expansion formula** (two stride-2 convs): `audio_len = ((L−1)//2 + 1 − 2)//2 + 1`; L=200 → 50, L=100 → 25, L≤2 → 0.
2. **Text runs** (before / between / after audio blocks): all three rows get `arange(text_len) + st_idx`.
3. **Audio block**: bos (1) + body (`audio_len`) + eos (1), each `arange(n) + st_idx`, identical rows.
4. **Segment chaining**: `st_idx = previous max + 1` (first segment 0) — hence the whole row is a contiguous arange.
5. **mrope delta** = `max + 1 − expanded length` (official `input_ids` are already expanded). For text+audio this is always 0; **non-zero delta only arises from image/video** (h×w tokens share one position id).
6. **Length conservation**: `N_expanded = len(input_ids) − 3·blocks + Σ(2 + audio_len)`.

## Constraints

1. Special tokens: TEXT=0, AUDIO_START=1, AUDIO_PAD=2, AUDIO_END=3; blocks must be `[START, PAD, END]`, no stray tokens.
2. Each `audio_seqlens` entry is an int ≥ 1; block count must match.
3. Empty `input_ids` → empty row, delta 0.
4. `audio_expand_len` is independently testable (L=1/2 → 0, L=3 → 1).
5. No input mutation; pure integer ledger, no floats.

## Data Scale

- Up to 200 input tokens, ≤ 3 audio blocks, mel length ≤ 10⁴ (audio_len ≤ 2500).

## Acceptance Anchors (excerpt)

- Pure text `[T×5]` → row `[0..4]`, delta 0.
- `T T [A(L=200)] T T T` (8 input tokens) → expanded row `[0,1,2,3..52,53,54,55,56]` (57 slots), delta 0.
- Two blocks `T [A(L=40)] T T [A(L=8)] T` → 20 slots, delta 0.
