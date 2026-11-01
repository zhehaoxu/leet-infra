# Qwen2.5-Omni Thinker-Talker 隐态配比账

## 题意

transformers `modeling_qwen2_5_omni.py`（v4.52）中 Talker 把 thinker（文本 LLM）的
每 token 隐态逐帧回灌进音频 LM。维度压缩为 D=1，复现完整数据流账（神经网络剥离，
隐态/embedding 全作标量入参）：

1. **prefill**：`talker_embed[i] = prompt_hiddens[i] + prompt_embeds[i]`，但
   `audio_mask[i]=True` 的位置 embedding 先置 0（**只零化 embedding，隐态保留**，
   官方 masked_scatter_ 语义）；末尾追加 `[text_bos_embed, reply[0]]` 两帧，
   其中 `reply[i] = thinker_hiddens[i] + thinker_embeds[i]`。
2. **input_ids 模式**：`[codec_mask]*L + [codec_pad, codec_bos]`（输出编码
   mask=-1, pad=-2, bos=-3）。
3. **reply 队列**：初始 = `reply[1:] + [eos_embed, pad_embed]`，长 T+1——eos/pad
   是文本 eos 之后两步 decode 的兜底帧（官方约束 n_decode ≤ T+1，空队列即崩）。
4. **decode 1:1**：每步 `summed = codec_embeds[codec_tokens[t]] + queue[0]`，
   消费后砍头。每生成 1 个 codec token 恰消费 1 帧 thinker 隐态（配比 1:1）。

## 接口

```python
solve(prompt_hiddens, prompt_embeds, audio_mask,
      thinker_hiddens, thinker_embeds,
      text_bos_embed, eos_embed, pad_embed,
      codec_embeds, codec_tokens, n_decode)
```

返回：

```python
{
  "prefill_embeds": list,       # 长 L+2
  "prefill_input_ids": list,    # [-1]*L + [-2, -3]
  "initial_reply_queue": list,  # reply[1:] + [eos, pad]
  "steps": list,                # 每步 [summed, consumed_idx]
  "final_reply_len": int,       # T + 1 - n_decode
}
```

约束：三输入等长；T ≥ 1；`n_decode ≤ T + 1`；codec_embeds 覆盖所用 token。

## 示例

```python
solve([1.0, 2.0], [10.0, 20.0], [False, True],
      [100.0, 200.0], [1.0, 2.0],
      0.5, 0.1, 0.2, {5: 0.05, 6: 0.06}, [5, 6, 5], 3)
# prefill: audio 位置 embedding 置零 → [11.0, 2.0]，尾部 [0.5, 101.0]
# -> prefill_embeds=[11.0, 2.0, 0.5, 101.0], prefill_input_ids=[-1,-1,-2,-3]
#    initial_reply_queue=[202.0, 0.1, 0.2]
#    steps=[[202.05, 0], [0.16, 1], [0.25, 2]], final_reply_len=0
```

## 数据规模与判题

- L, T ≤ 6；300 组随机对拍；变异 14 条全抓（相加语义、audio 只零化 embedding、
  尾部两帧与顺序、队列含兜底帧、decode 砍头、ids 编码、reply 相加公式等）。
- 浮点锚点用 `math.isclose` 比较。
