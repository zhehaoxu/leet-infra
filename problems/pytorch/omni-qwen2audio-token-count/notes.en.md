# Notes: omni-qwen2audio-token-count

## Source

HF transformers `processing_qwen2_audio.py` (per-item feature_attention_mask, `<|AUDIO|>` placeholder) + `modeling_qwen2_audio.py` `Qwen2AudioEncoder._get_feat_extract_output_lengths`:

```python
input_lengths = (input_lengths - 1) // 2 + 1
output_lengths = (input_lengths - 2) // 2 + 1
```

Expansion happens in model forward (mask sum → conv formula → placeholder replacement).

## Key Judgments

- **Pool first, then conv**: `sum(mask)` feeds the formula; `len(mask_row)` must not — padding invariance is a standalone test.
- **audio_len=0 is legal**: mel ≤ 2 frames get crushed by two stride-2 convs; the placeholder is replaced by nothing. Pinned via `expanded_len = len − 1` "net removal" semantics.
- **Mapping is by audio entry order, not placeholder appearance order**: the processor fills features per audio item; the placeholder is just a marker. Pinned by `check_multi_audio_order` (shuffled placeholders + ordered mask rows).
- Hand-anchor slip: I guessed L=5/6 → 2, actual 1. Lesson: evaluate every conv layer numerically, never by feel.

## Boundary

- No real mel/whisper encoder; no conv weights or numeric forward (framework-operator territory).
- Split with 674 `omni-tmrope-position`: that one computes *where* expanded positions go; this one computes *how many* tokens. Same conv structure, mutually referential.
- No video/audio-in-video interleaving (qwen2_5_omni scope).

## Test Design

- Oracle `_py` replays independently (per-row for-loop sum, two-step conv written out).
- 8 checks: anchor, padding invariance, conv boundary table (8 tiers), audio_len=0 net removal, multi-audio order, 150 random comparisons, 6 invalid-input tiers.
- Mutations 9/9: each conv ±1 / floor swap, skipping conv, placeholder count ±1, expanded not subtracting placeholders, no pooling (len(row)), pooled≥1 unchecked, mask values unchecked, token id relaxed. `zero-allowed` survived initially (count check raised first); adding a "placeholder count matches but contains id=1" case caught it.
