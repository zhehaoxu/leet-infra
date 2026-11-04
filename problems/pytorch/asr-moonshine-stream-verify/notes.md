# 解题笔记 — Moonshine 流式投机验证

## 官方账

- moonshine-ai/moonshine `core/moonshine-streaming-model.cpp`（encode / compute_cross_kv / decode_full / decoder_reset）+ `.h`（config 常量：frame_len=80、total_lookahead=16、左上下文 16×depth、max_seq_len=448）；官方注释明示 biaser 走 teacher-forced 前缀与 rewind 的原因。
- 判题域：稳定帧/滑窗账、解码预算、cross-KV 生命周期、greedy 投机验证分歧点 + rewind + biaser 交互。前端 ORT 增量、词时间戳、Trie 构建（=826）不入题；spec 合集的采样式验证不重叠（本题 greedy 无采样）。

## 踩坑

- **变异锚点撞 docstring（820 坑第二次重演）**：`left = 16 * depth`、`window_start = max(...)`、`cache_reset = True` 都先出现在模块 docstring 里，replace 落在注释上变异无效假幸存——锚点必须带缩进上下文。
- **两个真等价剔除**：① stable 去掉 `max(0,·)`（负 stable 被 `new_frames <= 0` 兜底）；② 验证期 biaser walk 走 predictions 与走 teacher-forced 前缀——分歧即 rewind、全收时两者相等，**输出不可观测**（官方注释的正确性论据在输出层面无法变异判别）。
- **稳定帧断言自身算错**：total=30/emitted=14/lookahead=16 时 stable=14、new=0——要 total=46 才有 16 新帧；写用例时按 `stable = total − lookahead` 逐步推。
- **biaser trie 测试数据节点数一致**：depths 有 N 项 children 就得有 N 项（叶节点也要空 dict），IndexError 兜底会伪装成 CAUGHT-ERR。
- **rewind 的可观测性**：只在「分歧 + 有 biaser + rerun 行的 rival 在有无 bonus 间翻转」时可见——B3 用例专门构造（4: 1.0+3.386 vs 7: 2.0）；无 biaser 分歧测不到 rewind。
- **提案去重 max 的可观测性**：需要同一 token 被 root 与深节点同时提案且 rival 卡在两值之间（G 用例：3 = 0.5 + max(2, 3.386) = 3.886 < 4.0 拒收；叠加 5.886 则全收）。
- **EOS 永不入 tokens**：continue_ar 的 while 条件先判 eos；起点即 eos → 空结果；AR 行给 eos → 停止不追加。
- cache_policy 的 `("encode", 0)` 是 no-op 不是 error（early return 在 invalidate 前）——初版把无帧 encode 落进 unknown-op 分支被用例抓住。

## mut 17 变异清单

stable：去 is_final 全收、new≤0 夹取改 abs；window：left 16×→16+、去 0 夹取、size 用 emitted；max_tokens：ceil→floor、去 cap；cache：无帧 encode 也失效、decoder_reset 误失效；diverge：point 少 1、不 break（取最后匹配）；verify：去 rewind、cache_reset 恒 False；AR：eos 入结果、预算 <→≤；bias：叠加、深度用 parent。17/17 全 CAUGHT + 2 真等价剔除（stable-clamp、predwalk）。
