# Qwen2.5-Omni Thinker-Talker 隐态配比账

## 背景与回源

transformers `modeling_qwen2_5_omni.py`（v4.52）的 Talker 是"文本 LLM 隐态逐帧回灌
音频 LM"的全库唯一机制。官方 `generate` 中：thinker 先生成全部文本回复 token，其
每层隐态被缓存为 `thinker_hidden_states` / `thinker_token_embeds` 两条平行流；
Talker 的每一步 decode 都从这条 reply 队列取头帧。本题剥离神经网络，维度压成
D=1，考完整数据流账。

## 机制账

1. **prefill 相加而非拼接**：`talker_embed = thinker_hidden + token_embed` 逐元素
   相加（prefill 每个位置、reply 每帧都是这个公式）。
2. **audio token 只零化 embedding**：`embeds_to_talker` 在 audio/image/video 占位
   位置 `masked_scatter_` 置 0——embedding 没了但隐态保留（不是整帧丢弃）。
3. **prefill 尾部两帧**：`[text_bos_embed, reply[0]]`；对应 input_ids 模式
   `[codec_mask]*L + [codec_pad, codec_bos]`。
4. **reply 队列与兜底**：初始队列 = `reply[1:] + [eos_embed, pad_embed]`，长 T+1。
   eos/pad 两帧是给 thinker 文本 eos 之后的两步 decode 兜底——官方约束一次
   回复的 decode 步数 ≤ T+1，队列空即崩（官方无防御）。
5. **decode 1:1 配比**：每步 `summed = codec_embed(prev_token) + queue[0]`，消费后
   砍头。每生成 1 个 codec token 恰消费 1 帧 thinker 隐态——这是"流式节流"的
   实际口径（配比 1:1，无批量消费）。

## 判题要点

- 9 条固定锚点 + 300 组随机对拍；变异 14 条全抓（相加→取隐态、audio 整帧丢/
  隐态归零、尾部单帧、队首含 reply[0]、无兜底、eos/pad 换序、decode 不砍头、
  ids 模式换序/全 mask、reply 不加 embed、final_len 常量化）。
- 浮点锚点用 `math.isclose`（0.01+0.1 的二进制表示差异）。
