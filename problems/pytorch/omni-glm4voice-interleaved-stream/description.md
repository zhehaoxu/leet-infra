# GLM-4-Voice 语音-文本交错流解析账

## 题意

GLM-4-Voice（zai-org/GLM-4-Voice `web_demo.py`）的助手输出是文本 token 与音频
codec token（`<|audio_N|>`）的交错流，demo 端逐 token 解析并按渐进块大小流式送
TTS。逐字复现解析循环（源码顺序敏感，quirk 全保留）：

每收到一个 token 按以下精确顺序处理：
1. `token_id == end_token_id` → `is_finalize = True`；
2. **flush 判定先于当前 token 分类**：`len(audio_tokens) >= block_size` 或
   （`is_finalize` 且 audio 非空）→ 发块（内容、当时 block_size、finalize 标志），
   块大小阶梯 `block_size_idx` 未到末尾则 +1，audio_tokens 清空。end 且 audio 空
   时不 flush；
3. `not is_finalize` 才分类——**end 之后所有 token 停止分类**（complete_tokens
   不再增长）；音频值 = `token_id - audio_offset`（官方无上界检查）。

流结束后**没有最终 flush**：残留未满块音频静默丢弃（官方 quirk）。

## 接口

```python
solve(token_ids, audio_offset, end_token_id, block_size_list)
```

返回：

```python
{
  "complete_tokens": list,   # finalize 前分类过的全部 token id
  "text_tokens": list,       # 全部非音频 token id
  "chunks": list,            # 每块 [audio_values, block_size_at_flush, is_finalize]
  "dropped_tail": list,      # 流结束后残留未 flush 的音频值
  "final_block_size": int,   # 流结束时块大小（含升级）
  "is_finalize": bool,
}
```

约束：block_size_list 非空且严格递增；audio_offset > max(text token id)。

## 示例

```python
solve([100, 200, 201, 202, 300], 200, 300, [25, 50, 100])
# -> {"complete_tokens":[100,200,201,202], "text_tokens":[100],
#     "chunks":[[[0,1,2],25,True]], "dropped_tail":[], "final_block_size":25,
#     "is_finalize":True}

solve(list(range(200, 230)), 200, 999, [25, 50, 100])
# 第 26 个 audio 前 len=25>=25 触发 flush（判定先于分类）
# -> chunks=[[[0..24],25,False]], dropped_tail=[25,26,27,28,29],
#    final_block_size=50, is_finalize=False
```

## 数据规模与判题

- 流长 ≤ 530；400 组随机对拍；变异 14 条全抓（判定顺序、`>=`→`>`、去 offset 减、
  end 忽略、finalize 无条件 flush、阶梯升级、不清空、end 后继续分类、尾部保留、
  标志/尺寸常量化等）。
