# Moonshine 流式投机验证与缓存账

Moonshine（moonshine-ai/moonshine）的流式转写器（`core/moonshine-streaming-model.cpp`）把「音频增量进来 → 编码器只提交稳定帧 → memory 变化使 cross-KV 失效 → 贪心投机验证上一轮草稿 → 分歧点回退续解码」做成一条**跨 chunk 的状态机账**。与 spec 合集的采样式验证（rejection sampling / 残差 / 树验证）划界：这里是 **greedy argmax 验证**，无采样；与 826（biaser Trie 本体）划界：本题用 biaser 的走账语义但不考 Trie 构建。

## 机制拆解

### 1. 稳定帧账（encode）

- 编码器带 `total_lookahead = 16` 帧**前视**：非 final 只提交 `max(0, total − 16)` 帧「稳定」特征；`is_final` 全量提交；
- `new_frames = stable − 已提交`，≤ 0 不发（音频不足时 lookahead 吞掉全部）；
- 编码器滑窗重跑：每层固定左上下文 16 帧 → `left = 16 × depth`，`window_start = max(0, emitted − left)`，`window_size = total − window_start`；新帧从窗口内 `start_idx = emitted − window_start` 切出，adapter 以 `pos_offset` 续位，memory 追加。

### 2. cross-KV 生命周期（cache_policy）

- memory 追加（encode 产出 > 0 帧）→ `cross_kv_valid = False`；
- `compute_cross_kv` 从 memory 重建 → True；
- **`decoder_reset` 只清 self-KV，保留 cross-KV**（memory 未变）——官方注释明示；
- **无帧 encode 不碰标志**（early return 在 invalidate 之前）。

### 3. 解码预算

`max_tokens = min(ceil(memory_len × 0.020 × 6.5), 448)`——每帧 20 ms、语速 6.5 token/秒、封顶 max_seq_len。

### 4. 贪心投机验证（verify_decode）

- 验证：对 `[BOS] + draft` 的 logits **逐行 argmax**（teacher forcing），biaser 加成在每次 argmax **之前**；
- **biaser 走 teacher-forced 前缀**：`advance(tokens_with_bos[t+1])` 用草稿 token 而非预测——位置 t 条件于前缀本身（官方注释明示的坑）；
- `diverge_point` = 前缀连续匹配数（首个不匹配即停）；
- **全收**：从 `predictions[len(draft)]`（bonus 行）续 AR 解码；
- **分歧**：`cache_reset = True`（self-KV 清零重跑）、**biaser 回退到已接受前缀**（reset + 重走 accepted——此时 walk 还停在整个草稿上，含被拒 token）、取 `rerun_logits[diverge_point]` 行的加成 argmax 续解码；
- AR 循环：push 当前 token（**EOS 不入结果**）→ biaser advance → 下一行 logits argmax；EOS 或 `max_tokens` 预算截断停。

### 5. 与 826 的交互坑

- 验证期 bonus 翻转验收： rivals 更强时，深度对数爬坡的加成让草稿 token 被接受——**bonus 是否叠加（max vs sum）直接改变验收结果**；
- rewind 是否执行直接改变 rerun 行的 argmax（走账停在整个草稿 vs 已接受前缀）。

## 判题域

单入口 `solve(mode, **kw)`，六个 mode：`stable_frames` / `encoder_window` / `max_tokens` / `cache_policy` / `diverge_point` / `verify_decode`。logits 全显式入参、greedy argmax 无采样；前端 mel/conv 增量（ORT 图内部状态）、词时间戳、context biaser 的 Trie 构建不在域内。
