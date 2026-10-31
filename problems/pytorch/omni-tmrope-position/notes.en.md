# Notes: omni-tmrope-position

## Source

HF transformers `models/qwen2_5_omni/modeling_qwen2_5_omni.py` → `get_rope_index` (fetched via api.github.com contents + base64; `modeling_qwen2_5_omni_thinker.py` does not exist — TMRoPE lives in the main file). The audio expansion formula appears three times in-file (standalone blocks and the use_audio_in_video branch).

## Key Judgments

- **Delta denominator is the expanded length**: official forward expands `audio_pad` before calling `get_rope_index`, so the `len(input_ids)` in the delta formula is the expanded length. First anchor draft used the unexpanded length (49) — wrong; expanded gives exactly 0.
- **Text+audio delta is always 0** is the pinned counter-intuitive anchor: non-zero deltas only come from vision tokens sharing t/h/w positions.
- **Formula boundaries**: L=1/2 → 0 (two stride-2 convs crush 1–2 frames away), L=3 → 1. Pinned via a dedicated formula test, not only end-to-end.
- Official audio block is `audio_start + 1×pad + audio_end`; expansion happens in forward embedding, not in the processor.

## Boundary

- **Not covered**: the `use_audio_in_video=True` chunked interleaving (`get_chunked_index` slicing video/audio by `position_id_per_seconds × seconds_per_chunk`) — involves video grids and fps; left as a separate candidate.
- **Not covered**: image/video 3-axis grids (`get_llm_pos_ids_for_vision`); bounded against `mm-qwen25vl-window-index`.
- Split with 675 `omni-qwen2audio-token-count`: that one computes *how many* tokens to expand (feature-mask pooling + conv formula); this one computes *where* the expanded positions go.

## Test Design

- Oracle `_py` replays independently: per-slot `base = max(out)+1`, no reuse of the solution's segment structure; random cases cover 0–3 blocks, mel ∈ {1,2,3,8,25,40,100,200}.
- Three hand anchors: pure text, single block L=200, two blocks L=40/8 (with the length-conservation assert).
- Mutations 9/9: formula ±1 (each conv), dropped bos/eos, audio_len = raw mel length, text runs without offset, chaining −1, delta missing +1, single-row return.
