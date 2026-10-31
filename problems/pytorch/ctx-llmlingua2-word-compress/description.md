# 上下文压缩：LLMLingua-2 词级保留概率合并与阈值

## 题目描述

LLMLingua-2（2403.12968）用一个 BERT 级编码器给每个 token 打二分类标签，`softmax` 后取「保留」类概率，再按词做合并、按目标压缩率取阈值，删掉低概率内容。真正决定压缩结果的是后处理：哪些子词属于同一个词、force token 如何强制保留、连续重复的强制词如何处理、阈值在词级还是 token 级上取。本题不加载模型，直接输入 token 串与每 token 的 keep 概率，复现这套后处理。

## 任务

实现 `solve(tokens, token_probs, reduce_rate=0.5, token_to_word="mean", force_tokens=None, token_map=None, force_reserve_digit=False, drop_consecutive=False, special_tokens=None, model_name="xlm-roberta-large", token_count=None)`：

- `tokens`：`list[str]`，逐个 token 的字符串；
- `token_probs`：`list[float]`，与 `tokens` 等长，每个 token 的 keep 概率，取值在 `[0, 1]`；
- `reduce_rate`：`0 < rate < 1`，目标删除比例；
- `token_to_word`：`"mean"`（子词概率取均值）或 `"first"`（取首个子词）；
- `force_tokens`：无论概率多少都保留的词（按纯词形匹配）；
- `token_map`：`{原词: 新增 token}`，新增 token 在合并时替换回原词；
- `force_reserve_digit`：为真时，含数字的 token 概率强制为 `1.0`；
- `drop_consecutive`：为真时，连续重复的 force 词只保留第一次；
- `special_tokens`：直接跳过、不参与词表；
- `model_name`：只支持 `"xlm-roberta-large"`（`▁` 开头新词）与 `"bert-base-multilingual-cased"`（`##` 续接）；
- `token_count`：可调用对象 `word -> int`，给出该词在目标分词器中的 token 数；`None` 视为每个词 1 个 token。

返回 `dict`，键为 `words`、`word_probs`、`keep_words`、`keep_labels`、`threshold`。

## 规则

1. 遇到 `special_tokens` 中的 token 直接跳过。
2. 判断词首：`xlm-roberta-large` 下，token 是标点、在 `force_tokens` / `token_map.values()` 中，或以 `▁` 开头；`bert-base-multilingual-cased` 下，去 `##` 后命中 force/map，或不以 `##` 开头。
3. 词首 token 若纯词形命中 `force_tokens` 或 `token_map.values()`，其概率置 `1.0`；随后按 `token_map` 把替换 token 还原成原词。
4. 续接 token 去掉 `##` / `▁` 后拼到上一个词；若 `force_reserve_digit` 且 token 含数字，其概率置 `1.0`。
5. 词概率按 `token_to_word` 的 `mean` 或 `first` 合并。
6. `drop_consecutive=True` 时，先取 `percentile(word_probs, int(100*rate))` 作为词间阈值；对每个 force 词，若它前面隔着「高于词间阈值」的非 force 词则重置状态，否则若与前一个 force 词相同就把概率置 `0`。
7. 把每个词概率按 `token_count(word)` 展开成 token 级列表，取 `threshold = percentile(expanded, int(100*rate + 1))`（默认线性插值，`+1` 再截断）。
8. 保留 `p > threshold`，或 `threshold == 1.0` 时保留 `p == threshold`；`drop_consecutive` 下若当前 force 词与已保留的最后一个词相同，则标签记 `0`。

## 示例

```
tokens = ["▁Hello", "▁world", "▁Hello", "▁world"]
token_probs = [0.9, 0.4, 0.8, 0.3]
reduce_rate = 0.5

words       = ["▁Hello", "▁world", "▁Hello", "▁world"]
word_probs  = [0.9, 0.4, 0.8, 0.3]
threshold   = 0.612
keep_words  = ["▁Hello", "▁Hello"]
keep_labels = [1, 0, 1, 0]
```

## 约束

- `tokens` 与 `token_probs` 非空且等长；概率必须是 `[0, 1]` 内的实数。
- 第一个非特殊 token 必须是词首，否则抛 `ValueError`。
- `reduce_rate` 必须在 `(0, 1)` 内；`model_name`、`token_to_word` 只接受列出的取值。
- `token_count` 必须返回 `>= 1` 的整数；`None` 时每个词按 1 个 token 计。
- 不修改输入列表；所有非法输入抛 `ValueError`。
