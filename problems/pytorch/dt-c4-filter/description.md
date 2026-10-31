# C4 行级清洗与去重：clean_page 过滤顺序账

## 题意

C4（T5 语料）的核心清洗是 `clean_page` 的**行级过滤顺序账**（HF datasets legacy `c4_utils.py` 为权威源）。本题逐行转写该顺序，并附上 C4 的行级 md5 去重（`remove_duplicate_text` 的单页内 + 跨页显式状态版）：给定页面文本与可选 badwords、已见行哈希，输出保留行、清洗文本、各过滤器的丢行计数（上游 Beam counter 的确定性对应物）与整页拒绝原因。

## 求解

每行 `strip()` 后按上游顺序：

1. 任一词长度 > `max_word_length`(1000) → 丢行；
2. 剥引用 `\[\d*\]|\[edit\]|\[citation needed\]`；
3. 不以 `.` `?` `!` `"` 结尾，或以 `...` 结尾 → 丢行；
4. 词数 < `min_words_per_line`(5) → 丢行；
5. `lorem ipsum` → **整页拒**；`javascript` → 丢行；`{` → **整页拒**；
6. policy 子串（terms of use / privacy policy / cookie policy / uses cookies / use of cookies / use cookies）→ 丢行；
7. badwords 正则 `[^a-z](bw1|bw2|…)[^a-z]` 命中 → **整页拒**；
8. 去重：`md5(line.strip().lower())` 已见 → 丢行，否则保留并入 seen；
9. 每有效行记 1 句（上游 NLTK punkt 的确定性代理，题面钉死），总数 < `min_num_sentences`(3) → **整页拒**。

关键点：badword 锚点正则**词首/词尾/词中嵌套都不命中**（如 `sex` 在 `essex` 内不触发，上游原样保留的 quirk）；去重键是 strip+lower 后的 md5（大小写/空白变体同键）；整页拒绝时保留已发生的丢行计数，`num_sentences` 记拒绝前的有效行数。

## 数据规模

- 页面按行处理，行数 ≤ 64；阈值可调但须正整数。
- text 非字符串、badwords 含空串、seen_hashes 类型错、阈值 ≤0/bool/浮点 → `ValueError`。
