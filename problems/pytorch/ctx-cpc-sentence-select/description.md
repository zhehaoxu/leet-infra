# Problem

**CPC**（Context-aware Prompt Compression，arXiv:2409.01227，AAAI-25，Workday）训练了一个双向注意力句编码器，用「问题向量 vs 句向量」的余弦相似度给每个句子打相关性分，然后按 token 预算**整句保留/丢弃**——避免 token 级删除破坏句子语义。

本题复现打分之后的确定性选择链。编码器打分（模型 forward + mean pooling + cos_sim）、tokenizer 计长、正则命中全部离线入参化：

1. **boost_flags**——`re.match` 语义（**仅句首**命中才算 boost，句中出现不算）；pattern 为 None 时全 False；
2. **rank_key**——`(boost, score)` **双键降序** stable 索引：boost 句全部排前（组内按分降序），其余按分降序，并列保原序；
3. **select_segments**——官方贪心：`taken` 从 **-1** 起（补偿 +1 的空格账），每句 `taken += ln + 1`；**溢出句先收再 break**（add-then-break 怪癖：让预算爆掉的那句照样入选）；`taken` 恰好等于预算时**继续**收；fake（question echo）句一律跳过；**输出按原句序过滤**（不是分数序）。

## 任务

实现 `solve(mode, **kw)`，支持三个模式：

- `boost_flags(segments, pattern)` → 布尔列表；
- `rank_key(boosts, scores)` → 降序 stable 索引列表；
- `select_segments(segments, scores, boosts, fakes, lengths, target)` → `(selected_segments, sorted_indices, taken)`。

输入均为列表；输出与锚点逐项一致。

## 数据范围与约定

- 句数 ≤ 32；分数为有限浮点；lengths 为各句 token 数（≥ 1）；
- `target` 为 token 预算（整数）；fake 句在排序中参与但选择时跳过；
- 并列 `(boost, score)` 完全相同的一律稳定排序（按原序）；
- 全部句子都是 fake 时返回空列表，`taken` 保持 -1。
