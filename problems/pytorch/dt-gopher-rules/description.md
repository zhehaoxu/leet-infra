# Gopher 规则质量过滤：九条启发式的顺序账

## 题目描述

Gopher（DeepMind, arXiv:2112.11446）在训练 280B 模型前用一套**规则启发式**清洗语料：
不跑模型、不学参数，只靠九条可解释的阈值规则把低质文档逐条筛掉。它至今仍是数据管线的
标配基线——HuggingFace 的 datatrove 库给了权威实现 `GopherQualityFilter`。规则简单，
**账难对齐**：每条规则的分子分母用什么口径（全部词还是非标点词？行级判定前要不要
strip？）、九条规则按什么顺序短路、边界值算过还是不过，任何一处口径差一点，最终留
下的语料就是另一份。

> 为保证判题**确定性**：分词钉死为 `text.split()` 空白切分（上游用 NLTK Treebank
> tokenizer，判题环境零依赖不可用）；标点集钉死为题面给出的 63 个字符（上游集合更大，
> 本题取 ASCII 全集 + 常见全角标点与省略号/破折号）；九条规则全部恒开。

## 规则

- 实现 `solve(text, min_doc_words=50, max_doc_words=100000, min_avg_word_length=3,
  max_avg_word_length=10, max_symbol_word_ratio=0.1, max_bullet_lines_ratio=0.9,
  max_ellipsis_lines_ratio=0.3, max_non_alpha_words_ratio=0.8, min_stop_words=2)`，
  按以下**顺序**检查，首个失败规则即短路（`failed_rule`），全部通过则 `passed=true`：

  1. `gopher_short_doc`：非标点词数 `< min_doc_words` 拒（`=` 通过）
  2. `gopher_long_doc`：非标点词数 `> max_doc_words` 拒
  3. `gopher_avg_word_length`：非标点词的**含标点字符**平均长度出界（`< min` 或 `> max`）拒
  4. `gopher_too_many_hashes`：`text.count("#") / n_words > max_symbol_word_ratio` 拒
     ——分母是**全部词数**（含纯标点词），非标点词才做分母是常见实现错误
  5. `gopher_too_many_ellipsis`：`（count("...") + count("…")) / n_words > max_symbol_word_ratio` 拒
  6. `gopher_too_many_bullets`：行 `lstrip()` 后以 `•` 或 `-` 开头的行占比 `> 0.9` 拒
  7. `gopher_too_many_end_ellipsis`：行 `rstrip()` 后以 `...` 或 `…` 结尾的行占比 `> 0.3` 拒
  8. `gopher_below_alpha_threshold`：含至少一个字母字符（`isalpha`）的词占比 `< 0.8` 拒
  9. `gopher_enough_stop_words`：与 8 个停用词 {the, be, to, of, and, that, have, with}
     的交集大小 `< 2` 拒

- 返回 dict：`passed`、`failed_rule`（首个失败规则的 reason 字符串或 None）、
  `checks`（九条规则名 → bool，按上述顺序）、账本 `n_words` / `n_non_symbol_words` /
  `avg_word_length` / `bullet_lines_ratio` / `end_ellipsis_lines_ratio` /
  `alpha_words_ratio` / `n_stop_words`。

## 示例

```
输入：text = "the model that we have is small and it works fine today"（12 词）
  min_doc_words=50 → 非标点词 12 < 50 → 首条短路
输出：passed=false, failed_rule="gopher_short_doc"，
      checks[短文档]=false（后续规则仍逐条算出但不短路）
```

## 约束

- 纯 Python（torch 无关）；空文本是合法输入（`n_words=0`，比率类账除法有 0 保护）
- 词的定义：`text.split()`；**纯标点词**（每个字符都在标点集内）不计入非标点词数，
  但计入 `n_words` 与占比类分母
- 行的定义：`text.splitlines()`；bullet 判定用 `lstrip()`、end-ellipsis 判定用
  `rstrip()`（前导空白后的 `-` 是 bullet，行尾空白前的 `...` 仍是 end ellipsis）
- 阈值边界一律**严格不等号判拒**：恰好等于阈值通过
