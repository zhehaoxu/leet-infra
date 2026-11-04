# Problem

**QUITO**（Query-gUIded aTtention cOmpression，arXiv:2408.00274，CCIR 2024）用一个小模型回答一个「触发问题」，拿问题对上下文各 token 的注意力分布给上下文打分，再按预算过滤低关注内容——压缩发生在主模型读入之前，主模型一个 token 都不用算。

本题复现打分之后的全部确定性过滤账。注意力分布（依赖 HF 前向：chat template、trigger token、最后一层注意力按头平均取末行再 softmax）作为离线入参直接给定，你实现的是从分数到压缩结果这条没有任何随机性的规则链：

1. **reconstruct**——子词聚合：`Ġ`/`▁` 前缀开启新词，续词拼入当前词并取 **max** 聚合注意力、cnt 自增；
2. **gaussian_smooth**——注意力序列的高斯平滑（scipy `gaussian_filter1d`，`mode='reflect'`、`truncate=4.0`，σ=0 恒等）；
3. **text_filter**——词级过滤：按分**降序**贪心，cnt 累积**严格小于**阈值 `sum(cnt)*ratio` 才保留，一旦放不下立即 break，输出保持**原序**；
4. **sentence_filter**——句级过滤：句分 = 句内词分 **max**，预算 `len(words)*ratio`，按分降序贪心、**大于**预算才断句（等额可过），选中句排序还原原序；
5. **dynamic_span**——动态句级：断点句内从最高分词出发**左右交替**扩展凑足剩余预算，还原原序。

## 任务

实现 `solve(mode, **kw)`，支持五个模式：

- `reconstruct(tokens, attention)` → `(words, attns, cnts)`；
- `gaussian_smooth(scores, sigma)` → 平滑后的列表；
- `text_filter(words, score, cnt, ratio)` → `(res_words, res_scores, keep)`；
- `sentence_filter(words, attention, sizes, ratio)` → `(selected_sentences, breakpoint_seg, budget_left)`；
- `dynamic_span(bp_words, bp_attn, m)` → `(selected_words, selected_attns)`。

输入均为列表；输出与锚点逐项一致（浮点容差 1e-9）。

## 数据范围与约定

- token 数 ≤ 64，词数 ≤ 32，句数 ≤ 8；分数为有限浮点；
- `cnt` 由聚合规则产生、恒 ≥ 1；`sizes` 各句词数之和等于词数；
- 断点句在「全部句子都放得下」时为 `None`，`budget_left` 亦为 `None`；
- 并列分数一律**稳定排序**（同分按原序），max 取首个最大值。
