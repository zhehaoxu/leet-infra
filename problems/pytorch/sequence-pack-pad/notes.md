# 序列打包（sequence packing）详解

## 为什么要打包

逐条 padding 时，一个 batch 里若最长 512、多数只有几十 token，则绝大部分算力花在 pad 上（softmax、FFN 全白算）。打包把多条短序列**无缝**拼进固定长度块，pad 只出现在最后一块尾部，token 利用率从「平均长度/最大长度」一举逼近 ~100%。Megatron、NeMo、llm.c、多数 SFT 框架默认开启。

## position_ids 是打包的灵魂

普通 padding 里 `position_ids` 就是 `arange(L)`。打包后一条块内含多条子序列，位置必须**每份子序列各自从 0 重数**，否则第二份子序列会被赋予「接着上一条」的位置——RoPE / 位置嵌入直接错位。所以 `position_ids` 记录的是「在本原子序列内的下标」，与块边界无关：一条子序列跨越块边界时，位置**继续递增**，不重置（重置只发生在「新子序列开始」）。这是本题最容易写错的地方。

## segment_ids：让注意力不越界

块内相邻的两份子序列语义无关，若让它们互相 attend 就是「文档串味」。判据：注意力只允许在**同一 segment**内。下游用 `segment_ids[i] == segment_ids[j]` 生成 4D block-diagonal mask（再叠加 causal）。本题只需把 `segment_ids` 产出，pad 用 `-1` 与所有真实子序列区分。

- 用 `segment_ids == -1` 作为 pad 的唯一判据，`attention_mask` 也由它导出——避免两套 pad 逻辑打架。
- 子序列序号跨块**连续**（第 3 条就是 3，不管它落在第几块），这样同一子序列跨块部分仍共享 id。

## 与 data-collator 的取舍

| | data-collator | sequence-pack-pad |
|---|---|---|
| pad 位置 | 每条尾都要补 | 只补最后一块 |
| 额外产出 | attention_mask, labels | position_ids, segment_ids |
| 注意力 | 只需 causal+右 pad mask | 还要 block-diagonal 防跨序列 |
| 利用率 | 低 | 高 |

## 关联

- 上游：`bpe-encoding` + `data-collator`（先转 id，再决定逐条 pad 还是打包）。
- 下游：打包后的 mask/position 喂给 `llm-rope` / attention 类题；与训练吞吐主题呼应。
- 判题：整数张量、无随机；`check_example` 用 dump 的四矩阵，另有单序列整除、跨块递增、无 pad 用例。
