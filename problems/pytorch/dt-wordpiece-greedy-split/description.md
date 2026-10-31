# WordPiece 贪心最长匹配与词级 UNK

## 任务

实现 `solve`：给一段原始文本和一份 WordPiece 词表，按 BERT 系的口径切出子词序列，并产出 token id、码点偏移与「片段属于哪个词」的映射。

WordPiece 是 BPE/Unigram 之外第三条主流子词路线，它的实现只有几十行，但面试里答全的人不多，因为四个口径同时才对：**(a)** 每个位置取的是**最长**命中，不是「最短片段」也不是「最少片段数」；**(b)** 续接片段必须**带上续接前缀**（BERT 是 `##`）去查表，词表里那个光秃秃的 `cat` 在词中间永远用不上；**(c)** 一旦某个位置无任何命中，**整个词**塌成一个 `[UNK]`，之前已经切出来的片段全部作废——不是逐字 `[UNK]`；**(d)** 词长超过上限的词直接整词 `[UNK]`，不再尝试切分。判题额外要求算出码点偏移和 word→piece 映射，这正是下游做 word-level 任务（NER、span 抽取、对齐标注）时真正要用的两列。

## 输入

- `text`：`str`，原始文本，$0 \le |\mathit{text}| \le 4000$（码点）。
- `vocab`：`list[str]`，$1 \le |\mathit{vocab}| \le 20000$，下标即 token id；每项非空且互不相同。
- `unk_token`：`str`，必须出现在 `vocab` 里。
- `continuation_prefix`：`str`，可以是空串，也可以多字符。
- `max_input_chars_per_word`：Python `int`（`bool` 不算），$\ge 1$。

预分词只有一条规则：`text` 按 Python 的空白定义切词（连续的 `str.isspace()` 字符作分隔，首尾空白不产生空词），词序即 word 编号 $0,1,2,\dots$。不做小写化、不做去音符、不做标点拆分、不认 special token。不修改输入，非法输入抛 `ValueError`。

## 精确规则

对第 $w$ 个词 $t$（码点数 $|t|$，在 `text` 中的全局半开区间 $[a,b)$）：

**超长短路**：若 $|t| > \mathit{max\_input\_chars\_per\_word}$，该词只产出一个片段 $\mathit{unk\_token}$，偏移 $[a,b)$，`word_id` 为 $w$。不再尝试任何切分。

**贪心最长匹配**：否则令 $\mathit{start}=0$，反复执行

$$
\mathit{piece}(\mathit{start})=\arg\max_{\mathit{end}\,>\,\mathit{start}}\
\bigl\{\,|\mathit{cand}(\mathit{start},\mathit{end})| : \mathit{cand}(\mathit{start},\mathit{end})\in \mathit{vocab}\,\bigr\},
\qquad
\mathit{cand}(\mathit{start},\mathit{end})=
\begin{cases}
t[\mathit{start}:\mathit{end}], & \mathit{start}=0,\\
\mathit{prefix}+t[\mathit{start}:\mathit{end}], & \mathit{start}>0,
\end{cases}
$$

命中后把该片段原样（含前缀）写入输出，$\mathit{start}\leftarrow\mathit{end}$，直到 $\mathit{start}=|t|$。

**词级 UNK**：若某个 $\mathit{start}$ 上不存在任何命中，则**丢弃该词已产出的全部片段**，只留一个 $\mathit{unk\_token}$，偏移 $[a,b)$，`word_id` 为 $w$。注意这等价于「一个词要么被完整切成词表里的片段，要么整词只有一个 `[UNK]`」，输出里不会出现「半个词若干片段 + 一个 `[UNK]`」的混合。

**贪心不是搜索**：最长匹配不保证片段数最少，也不保证「存在某种切分」时一定能切成。词表里有 `cat` 但续接位需要 `##cat`，而 `##cat` 不在词表时，即使 `ha`/`##sh`/`cat` 三个串都能覆盖整词，结果仍是整词 `[UNK]`。

## 返回

`(tokens, ids, offsets, word_ids)` 四元组，长度都是片段总数 $P$（可以为 0）：

- `tokens`：`list[str]`，原样片段字符串（续接片段**带**前缀，`[UNK]` 就是 `unk_token` 本身）。
- `ids`：`list[int]`，每项是 `tokens[i]` 在 `vocab` 中的下标。
- `offsets`：`list[list[int]]`，每项 `[start, end)`，是 `text` 的**码点**下标（不是 UTF-8 字节，也不是 UTF-16 单元）。
- `word_ids`：`list[int]`，每项是该片段所属词的序号（$0$ 起）。

空文本或纯空白文本返回四个空列表。全部逐位比较。

## 示例

`vocab = ["[UNK]", "the", "un", "##af", "##fable", "bow", "##els", "a", "##a", "ha", "##sh", "cat", "##t"]`，`unk_token = "[UNK]"`，`continuation_prefix = "##"`，`max_input_chars_per_word = 10`，`text = "the unaffable bowels hashcat cat"`：

```python
# words      the(0,3) w0 | unaffable(4,13) w1 | bowels(14,20) w2 | hashcat(21,28) w3 | cat(29,32) w4
# tokens   ["the", "un", "##af", "##fable", "bow", "##els", "[UNK]", "cat"]
# ids      [    1,    2,     3,        4,     5,      6,       0,     11]
# offsets  [[0,3], [4,6], [6,8],   [8,13], [14,17], [17,20], [21,28], [29,32]]
# word_ids [   0,    1,     1,       1,      2,       2,       3,       4]
```

逐词读：`unaffable` 在 `start=0` 的最长命中是 `un`（`una`、`unaf`、…、`unaffable` 都不在词表），`start=2` 的最长命中是 `##af`（`##aff`、`##affa` 都不在），`start=4` 命中 `##fable`。`hashcat` 先拿到 `ha` 和 `##sh`，到 `start=4` 时只允许查 `##cat`、`##ca`、`##c`——词表里那个 `cat` 是首位置形态，用不上，于是整词塌成 `[UNK]`，前两个片段一并作废；紧随其后的独立单词 `cat` 因为 `start=0` 不加前缀而正常命中。把 `max_input_chars_per_word` 改成 3，同一条 `text` 里的 `bowels`（6 个码点）直接短路成 `[UNK]`，`offsets` 是 `[14, 20]`、`word_ids` 是 `2`。
