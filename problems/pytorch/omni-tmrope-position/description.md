# TMRoPE：文本+音频段的 3D 位置 id 账

## 背景与位置

Qwen2.5-Omni（听-看-说一体模型）thinker 的 LLM 用 **TMRoPE**：三轴（t/h/w）位置 id，视觉 token 沿三轴展开成 3D 网格，其余 token 三轴同值退化为 1D。HF transformers `modeling_qwen2_5_omni.py` 的 `get_rope_index` 就是这本账：按输入序扫特殊 token，逐段把位置 id 追加到 `llm_pos_ids_list`，最后拼成 `[3, bs, seq]`。

本题转写其中**文本 + 独立音频段**路径（`use_audio_in_video=False`、无图像/视频）：omni 的音频是"一等公民"——`<|audio_bos|> <audio_pad|> <|audio_eos|>` 三 token 块，模型 forward 时把中间 pad 展开成 `audio_len` 个音频 token，位置账随之逐段推进。

## 精确账目

1. **展开长度公式**（audio encoder 两层 conv，stride 2+2、kernel 各扣一帧）：
   `audio_len = ((L−1)//2 + 1 − 2)//2 + 1`，L 为池化后 mel 帧数。L=200 → 50、L=100 → 25、L≤2 → 0。
2. **文本 run**（含音频前的文本、块间文本、尾部文本）：三行同为 `arange(text_len) + st_idx`。
3. **音频块**：bos 1 个 + body `audio_len` 个 + eos 1 个，每段 `arange(n) + st_idx`，三行同值。
4. **段衔接**：`st_idx = 前段 max + 1`（首段 0）——文本与音频段都从"上一个位置 +1"继续，因此整条序列三行都是连续 `arange`。
5. **mrope delta**：`max + 1 − 展开后长度`（官方 `get_rope_index` 收到的 `input_ids` 已经展开）。文本+音频序列该值恒为 0；**非零 delta 只由图像/视频产生**（h×w token 共享同一位置 id，位置推进慢于 token 推进）。
6. **展开长度守恒**：`N_expanded = len(input_ids) − 3·块数 + Σ(2 + audio_len)`（每块的 3 token 换成 1+audio_len+1）。

## 约束

1. 特殊 token：TEXT=0、AUDIO_START=1、AUDIO_PAD=2、AUDIO_END=3；块结构必须 `[START, PAD, END]`，不允许游离 token。
2. 每条 `audio_seqlens` ≥ 1 的整型；块数与 `audio_seqlens` 数量一致。
3. 空 `input_ids` 返回空行、delta=0。
4. `audio_expand_len` 单独可测（公式边界 L=1/2 → 0、L=3 → 1）。
5. 禁改入参；全整型账，无浮点。

## 数据规模

- 序列长 ≤ 200 输入 token，音频块数 ≤ 3，单音频 mel 长 ≤ 10⁴（audio_len ≤ 2500）。
- 位置值本身不超过 10⁴，无需大数。

## 验收锚点（节选）

- 纯文本 `[T×5]` → 行 `[0..4]`、delta 0。
- `T T [A(L=200)] T T T`（8 输入）→ 展开行 `[0,1,2,3..52,53,54,55,56]`（57 位），delta 0。
- 双块 `T [A(L=40)] T T [A(L=8)] T`（10 输入）→ 展开 20 位，delta 0。
