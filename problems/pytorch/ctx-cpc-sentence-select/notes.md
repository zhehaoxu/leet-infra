# CPC 句级相关性选择 — 实现笔记

## 回源与口径

- 官方仓 `Workday/cpc`（AAAI-25，arXiv:2409.01227）。核心账在 `util/preprocessing.py::compress_sample`（`prompt_compressor.py` 只是壳）；编码器是训练出来的双向模型（mistral/llama LoRA，`LlamaBiForMNTPandSentEmbeddingsV2`），判题离线入参化。
- **离线入参映射**：句分 = cos_sim(句向量, mean_pooling(question 向量))；句 token 长 = tiktoken/GPT-4 tokenizer；boost = `re.match(boost_sents_regexp, seg)`；fake = `segments_labels==1`（question-as-suffix 拼进上下文的 echo 句）。本题直接给 `(segments, scores, boosts, fakes, lengths, target)`。
- **长度预算语义**：官方 `compression_target_tokens` 是 **token 预算**（不是压缩率）；`taken` 从 **-1** 起是官方对每句 `+1`（空格分隔）的补偿——所以预算的含义是「token 数 + 句间隔数」。

## 逐入口细节

### boost_flags
- `re.match` 只锚**句首**：`"xEND tail"` 不命中 `END`，`"END head"` 命中。官方 boost 语义 = 正则句（如指令句）无条件优先于所有普通句。
- pattern None → 全 False（官方 `re_checker` 里 `boost_match_regex is None` 直接 False）。

### rank_key
- `sorted(all_similarities, reverse=True, key=lambda x: (x[3], x[1]))`——**reverse=True 双键降序**：boost（True>False）最优先，然后分数。Python `sorted(reverse=True)` 是 **stable** 的：完全并列保原序（不要用取负分数的写法复刻——对 bool/负数会有坑，直接 reverse=True + tuple key 即可）。

### select_segments
- 官方逐字：`taken = -1`；`taken += len(openai_tokenizer.encode(seg)) + 1`；`indices.add(idx)`；`if taken > target: break`。
- **add-then-break 怪癖**：让 taken 第一次超过 target 的那句**照样入选**，然后停止——压缩结果可能超预算。
- **等额继续**：`taken == target` 时 `> ` 为假，继续收下一句（与 QUITO text_filter 的严格 `<` 相反、与 sentence_filter 的 `>` 一致）。
- **fake 跳过**：排序参与（占位）但选择时 `continue`——全 fake 时 indices 空、taken 保持 -1。
- **输出原序**：`[seg for ... in all_similarities if idx in indices]`——按原始 global_index 过滤，不是分数序。

## 真等价记录（勿再钉）

- **m-sl-breakfirst**：把 add-then-break 改成 break-then-add（先判 `taken > target` 再收句），**终态恒同**——溢出句被收时其前一时刻 taken 必然 ≤ target，两种顺序都会收下它并停止；选中集与 taken 逐位一致。变异无效，已剔除。

## 与 ctx 组其他题的分界

- QUITO（=801）：词/句两级**注意力分数**过滤 + 高斯平滑 + 动态中心扩展；本题 CPC 是**训练编码器分数** + boost 正则 + token 预算整句贪心，无平滑、无重排。
- LLMLingua-2：分类器打分的**词级**删除；LongLLMLingua：困惑度预算**按文档分配**；GemFilter：早层注意力 top-k。CPC 增量：boost 优先级、fake 跳过、`-1 起步 + ln+1` 的预算账、add-then-break。

## 验证链

- 沙箱 `/tmp/cpc_sandbox.py` 五段互证（boost/rank/select 基础 + fake/全 fake + 等额继续/溢出先收 + re.match 语义 + stable 并列）。
- selftest 双模式 5/5 ACCEPTED；mut802 **11/11 全 CAUGHT** + 1 真等价剔除（如上）。
- 干跑流程新惯例：护栏运行期间先在 /tmp 暂存并干跑全部验证（`mut802_dry` 指向暂存目录），护栏完成后再落地，避免半注册状态被全量扫描抓到（800 期 HpuENv 320 errors 的教训）。
