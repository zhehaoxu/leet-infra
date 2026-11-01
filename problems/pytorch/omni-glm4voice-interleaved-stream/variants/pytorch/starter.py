# GLM-4-Voice 语音-文本交错流解析账

GLM-4-Voice（zai-org/GLM-4-Voice `web_demo.py`）的助手输出是文本 token 与音频
codec token（`<|audio_N|>`）的**交错流**。demo 端逐 token 解析并按渐进块大小流式
送 TTS。本题逐字复现其解析循环（对齐源码顺序，quirk 全保留）。

## 接口

```
solve(token_ids, audio_offset, end_token_id, block_size_list)
```

- `token_ids`: list[int] —— 模型生成的 token id 流
- `audio_offset`: int —— `<|audio_0|>` 的 token id；`token_id >= audio_offset`
  即音频 token，值 = `token_id - audio_offset`（**官方无上界检查**）
- `end_token_id`: int —— `<|user|>` 的 id，作为终结符触发 finalize
- `block_size_list`: list[int] —— 流式 TTS 块大小阶梯（官方 [25,50,100,150,200]）

## 官方循环（逐条对齐源码，顺序敏感）

每收到一个 token，按以下**精确顺序**处理：

1. `if token_id == end_token_id: is_finalize = True`
2. **flush 判定（在当前 token 分类之前）**：
   `if len(audio_tokens) >= block_size or (is_finalize and audio_tokens):`
   发一块（audio_tokens 当前内容 + flush 时的 block_size + is_finalize 标志）；
   然后 `block_size_idx` 若未到阶梯末尾则 +1（块升级）；audio_tokens 清空。
   注意：end 且 audio 空时**不 flush**。
3. `if not is_finalize:` 才分类——end 之后**所有 token 停止分类**
   （complete_tokens 也不再增长）；否则 audio token 入 audio_tokens，
   其余入 text_tokens。

流结束后**没有最终 flush**：残留在 audio_tokens 里未满块的音频被静默丢弃
（官方 quirk）。is_finalize 与 flush 判定使得：end 时 audio 非空 → 立刻发
finalize 块；end 后即使还有音频 token 也不被解析。

## 输出 dict

```
{
  "complete_tokens": list,   # is_finalize 之前分类过的全部 token id（不含 end 及其后）
  "text_tokens": list,       # 全部非音频 token id
  "chunks": list,            # 每块 [audio_values, block_size_at_flush, is_finalize]
  "dropped_tail": list,      # 流结束后残留在 audio_tokens 的音频值（官方静默丢）
  "final_block_size": int,   # 流结束时的 block_size（含升级）
  "is_finalize": bool,
}
```

## 示例

```python
solve([100, 200, 201, 202, 300], 200, 300, [25, 50, 100])
# 100 → text；200/201/202 → audio 值 0/1/2；300=end：
#   flush 判定 (is_finalize and audio 非空) → 块 ([0,1,2], 25, True)
#   end 不分类；无后续
# -> {"complete_tokens":[100,200,201,202], "text_tokens":[100],
#     "chunks":[[[0,1,2],25,True]], "dropped_tail":[], "final_block_size":25,
#     "is_finalize":True}

solve(list(range(200, 230)), 200, 999, [25, 50, 100])
# 第 26 个 audio 到来前 flush 判定 len=25>=25 → 块 25 个，block 升级 50；
# 尾部 5 个未满块 → dropped_tail
# -> chunks=[[[0..24],25,False]], dropped_tail=[25,26,27,28,29],
#    final_block_size=50, is_finalize=False
```

## 约束

- `block_size_list` 非空且严格递增；audio_offset > max(text token id)。

写你的实现到 `solve`。
