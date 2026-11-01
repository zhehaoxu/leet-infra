# Qwen2.5-Omni Thinker→Talker 隐态配比账

transformers `modeling_qwen2_5_omni.py`（v4.52）中 Talker 的数据流是全库唯一的
"文本 LLM 隐态逐帧回灌音频 LM"机制。本题剥离所有神经网络，只考**确定性账**。
维度压缩为 D=1（embedding/隐态都是标量 float），相加语义不变。

## 输入

```
solve(prompt_hiddens, prompt_embeds, audio_mask,
      thinker_hiddens, thinker_embeds,
      text_bos_embed, eos_embed, pad_embed,
      codec_embeds, codec_tokens, n_decode)
```

- `prompt_hiddens` / `prompt_embeds`: list[float]，thinker prefill 最后一层隐态与
  输入 embedding（等长 L）
- `audio_mask`: list[bool]，长度 L——True 的位置是 audio 占位 token，其
  **embedding 在相加前被 masked_scatter 置 0**（官方 `embeds_to_talker` scatter）
- `thinker_hiddens` / `thinker_embeds`: list[float]，thinker 生成的 T 个回复
  token 的隐态与 embedding
- `text_bos_embed` / `eos_embed` / `pad_embed`: float
- `codec_embeds`: dict[int, float]，codec token embedding 表
- `codec_tokens`: list[int]，decode 阶段逐 token 输入（即上一步生成的 codec id）
- `n_decode`: int，解码步数（官方约束 ≤ T + 1：队列 = reply[1:] + [eos, pad] 共 T+1 帧）

## 官方口径（逐条对齐源码）

1. **prefill 输入 embeds**（送 Talker LM）：
   - 逐 token：`talker_embed = hiddens[i] + embeds[i]`，其中 audio_mask=True 的
     位置 `embeds[i]` 先置 0（隐态保留！官方只 scatter embedding）。
   - 末尾追加两帧：`[text_bos_embed, reply[0]]`，其中 reply[i] =
     `thinker_hiddens[i] + thinker_embeds[i]`。总长 L + 2。
   - 对应 input_ids 模式：`[codec_mask] * L + [codec_pad, codec_bos]`。
2. **reply 队列**：`reply[1:] + [eos_embed, pad_embed]`——**eos 和 pad 是队列尾部
   兜底两帧**（thinker 文本 eos 之后 Talker 还有两步可加）。
3. **decode 第 t 步**（t = 0..n_decode-1）：
   - `summed = codec_embeds[codec_tokens[t]] + queue[0]`（队列头帧与 codec embedding
     **逐元素相加，不是拼接**）
   - 消费后队列砍头：`queue = queue[1:]`。1:1 配比——每生成 1 个 codec token
     恰好消费 1 帧 thinker 隐态。
   - 记录 consumed 帧（相对初始 reply 队列的下标：T-1 起？——不，相对
     `[reply[1:]] + [eos, pad]` 的下标 0..T+1）。

## 输出 dict

```
{
  "prefill_embeds": list,        # 长 L+2
  "prefill_input_ids": list,     # [codec_mask]*L + [codec_pad, codec_bos]
                                 # （mask=-1, pad=-2, bos=-3 编码输出）
  "initial_reply_queue": list,   # reply[1:] + [eos, pad]，长 T+1
  "steps": list,                 # 每步 [summed, consumed_idx]
  "final_reply_len": int,        # T + 1 - n_decode
}
```

## 示例

```python
solve([1.0, 2.0], [10.0, 20.0], [False, True],
      [100.0, 200.0], [1.0, 2.0],
      0.5, 0.1, 0.2,
      {5: 0.05, 6: 0.06}, [5, 6, 5], 3)
# prefill: audio_mask[1]=True → embeds[1]=0 → embeds = [11.0, 2.0]
#          末尾追加 [0.5, 101.0] → [11.0, 2.0, 0.5, 101.0]
# input_ids: [-1, -1, -2, -3]
# reply queue: [202.0, 0.1, 0.2]
# steps: [[202.05, 0], [0.16, 1], [0.25, 2]]
# final_reply_len: 0
```

## 约束

- L = len(prompt_hiddens) = len(prompt_embeds) = len(audio_mask)；T ≥ 1；
  n_decode ≤ T + 1；codec_tokens 长度 ≥ n_decode；codec_embeds 覆盖所用 token。

写你的实现到 `solve`。
