# 出题笔记：omni-tmrope-position

## 回源

HF transformers `models/qwen2_5_omni/modeling_qwen2_5_omni.py` → `Qwen2_5OmniPreTrainedModelForConditionalGeneration.get_rope_index`（raw 拉取走 api.github.com contents + base64，`modeling_qwen2_5_omni_thinker.py` 不存在——TMRoPE 就在主文件）。音频展开公式同文件内三处出现（独立音频块、use_audio_in_video 分支各一次）。

## 关键口径判断

- **delta 分母是展开后长度**：官方 forward 先把 `audio_pad` 展开再调 `get_rope_index`，`mrope_position_deltas.append(llm_positions.max() + 1 - len(input_ids))` 里的 `input_ids` 已是展开序列。首版锚点按未展开长度算出 49，是错的；展开后恰为 0。
- **文本+音频 delta 恒 0** 是本题钉住的"反直觉锚点"：很多人以为多模态必有非零 delta，其实非零只来自视觉 token 的 t/h/w 共享。
- **audio_len 公式边界**：L=1/2 → 0（两层 stride-2 conv 对 1-2 帧输入压没），L=3 → 1。测试单独钉公式而非只钉端到端。
- 官方音频块是 `audio_start + 1×pad + audio_end`，pad 只占一个输入 token；展开发生在 forward embed 阶段，不在 processor。

## 边界划定

- **不做** `use_audio_in_video=True` 的 chunked 交错路径（`get_chunked_index` 按 `position_id_per_seconds × seconds_per_chunk` 切视频/音频交替块）——涉及视频 grid 与 fps 参数，账目复杂度数倍，留作独立候选。
- **不做**图像/视频三轴网格（`get_llm_pos_ids_for_vision`），与 `mm-qwen25vl-window-index`（多模态视觉侧）划界。
- 与 675 `omni-qwen2audio-token-count` 分工：那题算"展开多少个 token"（feature mask 池化 + conv 公式），本题算"展开后位置 id 怎么排"。

## 测试设计

- oracle `_py` 独立重放：逐位置 `base = max(out)+1` 追加，不复用 solution 的段结构；随机用例覆盖 0–3 块、mel ∈ {1,2,3,8,25,40,100,200}。
- 手算锚点三个：纯文本、单块 L=200、双块 L=40/8（含展开长度守恒断言）。
- 变异 9/9：公式 ±1（两层各一）、丢 bos/eos、audio_len 用 mel 原长、文本段不偏移、段衔接 −1、delta 少 +1、只返回单行。`row-single` 抓的是三行同值契约。
