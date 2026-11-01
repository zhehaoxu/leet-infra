# GLM-4-Voice 交错流解析账（notes）

回源：zai-org/GLM-4-Voice `web_demo.py`（模型主循环前段）。

- **分类判定**：`token_id >= audio_offset` 即音频（值 = id - offset），官方无上界
  检查——`<|audio_0|>` 之后的任何特殊 token id 都会被当音频，变异 `>=`→`>` 与
  去 offset 减均被锚点覆盖。
- **flush 与分类的顺序**：官方循环中 flush 判定在当前 token 分类**之前**——
  满 25 的块在"第 26 个音频到来时"才发出。变异把判定移到分类后即被锚点抓。
- **end 的三重效应**：置 is_finalize → 若 audio 非空立即发 finalize 块 →
  end 自身与其后 token 不进 complete_tokens（官方 `if not is_finalize` 包住分类）。
  end 且 audio 空时不 flush 也是官方行为（锚点 4）。
- **块阶梯**：`block_size_list = [25,50,100,150,200]`，flush 后升级，到顶不变；
  空阶梯（len=0）在官方会 IndexError，判题约定允许 assert/IndexError。
- **流结束无最终 flush**：残留 audio_tokens 静默丢弃——官方 quirk，输出以
  `dropped_tail` 显式呈现。
- 变异 14 条全抓；无等价变异。system_prompt 的交错节奏（"13 text token followed by
  26 audio tokens"）是提示词约束非解析账，未入题。
