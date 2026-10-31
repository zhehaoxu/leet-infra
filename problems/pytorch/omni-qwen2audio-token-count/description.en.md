# Qwen2-Audio: Audio Placeholder Token Expansion Ledger

## Background

In a Qwen2-Audio conversation, audio occupies a **single** `<|AUDIO|>` placeholder token; the model expands it into `audio_len` audio tokens during forward. The ledger lives in HF transformers `processing_qwen2_audio.py` + `modeling_qwen2_audio.py`: `feature_attention_mask` records pre-padding mel lengths, `_get_feat_extract_output_lengths` gives the two-conv output length.

## Exact Ledger

1. **Pooling kills padding**: `pooled_l = Σ feature_attention_mask[i]`; padding length has zero effect — the same audio padded to 400 or 100 frames yields identical results.
2. **Two-conv output length**: `audio_len = ((pooled_l − 1)//2 + 1 − 2)//2 + 1`; `pooled_l ≤ 2` → 0 (placeholder replaced by nothing, sequence shortens). Boundaries: L=1,2→0; L=3..6→1; L=7..10→2.
3. **Placeholder count conservation**: the number of audio placeholders in `input_ids` must equal the number of audios.
4. **Expanded length conservation**: `expanded_len = len(input_ids) − placeholders + Σ audio_len` (audio_len=0 nets −1 token).

## Constraints

1. `input_ids` allows only 0 (text) / 2 (audio placeholder); mask values only 0/1.
2. Each audio needs pooled ≥ 1.
3. Audios map to `audio_len` by **entry order** (mask row order), regardless of placeholder positions in `input_ids`.
4. No input mutation; pure integer ledger.

## Data Scale

- ≤ 4 audios, mel ≤ 10⁴ per audio, sequence ≤ 512, `expanded_len` ≤ 10⁴.

## Acceptance Anchors (excerpt)

- 200-frame audio + 56 pad, `[T,T,A,T,T]` → pooled 200, audio_len 50, expanded 54.
- mel=2 audio → audio_len 0, expanded_len = len − 1.
- Two audios `[A,T,A]`, mask [200 frames, 8 frames] → [50, 2], expanded = 3−2+52.
