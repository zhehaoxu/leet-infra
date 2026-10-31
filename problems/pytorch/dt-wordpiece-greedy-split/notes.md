# 参考思路：位置锚定的贪心最长匹配，失败按词结算

## 实现

三个循环嵌套，顺序不能换：

1. **切词**：`str.isspace()` 的极大连续段作分隔（与 `text.split()` 等价，参考解手工扫描、oracle 用 `split`，两条路互为对照）。词号按切出的顺序从 0 递增，UNK 不占号也不跳号。
2. **逐词贪心**：`start` 从 0 起，在每个 `start` 上把 `end` 从右往左收，第一个命中的就是最长命中。**关键是候选串**：`start == 0` 用裸串 `word[start:end]`，`start > 0` 必须用 `prefix + word[start:end]`。写成「全程裸串」或「全程带前缀」都错，而且两种错法在常见文本上都还能跑出看着合理的结果。
3. **按词结算**：任何一次 `start` 空手，就把这个词已经产出的片段全部丢掉，改成一个覆盖整词的 `unk_token`。所以 `offsets` 里 `[UNK]` 永远等于某个词的完整跨度；`tokens` 里也不会出现「片段 + 词中 UNK」的序列。

`max_span`（起始位取词表最长串长度、续接位再减前缀长度）只是把候选端点压小，更长的候选必然不在词表里，因此不改变结果——变异测试里「去掉这个上界」的 mutant 是等价的，判题抓不到它，也不该抓。写反方向（上界多减一）会漏掉真实命中，会被抓。

上游接口口径（离线源码逐字摘录）：

```text
# tokenizers 0.23.3, tokenizers/models/__init__.pyi:350-361
class WordPiece(Model):
    """
    An implementation of the WordPiece algorithm

    Args:
        vocab (:obj:`Dict[str, int]`, `optional`):
            A dictionary of string keys and their ids :obj:`{"am": 0,...}`

        unk_token (:obj:`str`, `optional`):
            The unknown token to be used by the model.

        max_input_chars_per_word (:obj:`int`, `optional`):
            The maximum number of characters to authorize in a single word.
    """

# tokenizers 0.23.3, tokenizers/implementations/bert_wordpiece.py
        wordpieces_prefix: str = "##",
    ):
        if vocab is not None:
            tokenizer = Tokenizer(WordPiece(vocab, unk_token=str(unk_token)))
        ...
        tokenizer.decoder = decoders.WordPiece(prefix=wordpieces_prefix)
```

实测默认值：`WordPiece(...).max_input_chars_per_word == 100`、`.continuing_subword_prefix == '##'`。

## 正确性检验

十条 `check_*`，全部与结构独立的 oracle 对拍（oracle 按首字符分桶、桶内按长度降序做 `word.startswith(rest, start)`，取第一个命中即最长；参考解是端点截串 + dict 查表，两条路径不共用中间结果）。除对拍之外还断言：

- `check_example`：题面例子的四元组逐位手算值，含上限改 3 的第二个分支。
- `check_matches_oracle`：24 组 fixture，覆盖多字符前缀、空前缀、`\t\n\u00a0` 混排、CJK、组合记号（`e` + U+0301）、增补平面 emoji、40 码点的长词、`unk_token` 不在词表首位（这条让「UNK 的 id 硬编码 0」无处可藏）。
- `check_word_level_unk_drops_pieces`：题面那个 `hashcat`（先命中 `ha`、`##sh`，第三格空手）必须只剩一个 `[UNK]`；并断言任何 `[UNK]` 都是它那个词的唯一片段、跨度恰等于词长——逐字 UNK 的答法在这里必炸。
- `check_greedy_is_not_minimum_pieces`：`ab/##c/##d` 三段对 `a/##bcd` 两段，贪心走三段；再给一个「贪心先命中长串所以一段搞定」的反向例子；最后 `abcd` + `ab/##c/a/##bcd` 断言贪心失败成 UNK，尽管 `a + ##bcd` 本可完整覆盖——这条同时打死「用 DP 求最少片段」的答法。
- `check_offsets_are_code_points`：增补平面字符按 1 个码点计（UTF-16 会是 2 个单元、UTF-8 会是 4 个字节），CJK 1 码点 3 字节；`_verify` 里另有 `text[start:end]` 与片段（去掉前缀后）逐字相等的断言。
- `check_prefix_forms`：空前缀时词中间可用裸词条；`###` 多字符前缀；词表里那条「恰好等于前缀」的 `##` 在 `start > 0` 永远命中不了、在 `start == 0` 是普通词条。
- `check_length_limit_boundary`：等于上限放行、超一个码点整词 UNK、上限为 1 的两个方向。
- `check_whitespace_and_numbering`：空串/纯空白返回四个空列表，词号连续。
- `check_invalid`：22 个非法入参 + 4 个合法边界。
- `check_maximum_shape`：词表 20000 项、文本 4000 码点以内、单个 3999 码点的词。

`_verify` 里还有两条通用不变量：**同词片段首尾相接且总长恰等于词长**、`ids == [vocab.index(t) for t in tokens]`，以及输入 `vocab` 未被修改。

变异测试 17 个 mutant：逐字 UNK、保留已切片段再补 UNK、全程裸串查表、命中带前缀但输出剥掉前缀、取最短命中、改成求最少片段的 DP、UTF-8 字节偏移、UTF-16 单元计数上限、上界多减一、词内局部偏移、`word_ids` 写成片段序号、UNK 的 id 硬编码 0，以及三条放宽校验（`max` 收 bool、词表允许重复、词表允许 tuple）、起始位错加前缀——**16 个被抓到**。唯一幸存的是「去掉 `max_span` 上界」，它与参考解逐位等价，只影响速度。

## 范围与来源

- **算法语义逐词对照过真实实现**：把参考解喂给 `.venv-eval` 里 tokenizers 0.23.3 的 `WordPiece`（配 `Whitespace` 预分词、无 normalizer、`add_special_tokens=False`），16 份词表 × 18 条文本、共 736 次逐词比对，**不含 `#` 的词 688 次全部一致**；48 次不一致全部落在词面含 `#` 的词上，实测如 `'a#' -> ['a', '[UNK]']`、`'##a' -> ['##', 'a']`，即上游 Rust 侧对前缀字符有额外分支（并且那里出现了逐字 UNK 的形态）。Rust 源码不在离线环境里，**本题不复现该分支**：含 `#` 的词按题面规则走，这条口径是题面自定。
- **题面自定、不做归属声明**的部分：码点偏移与 `word_ids` 这两个返回值（上游 `Encoding` 给的是它自己的 offset 语义，本题明确规定为 `text` 的码点下标）；`continuation_prefix` 作为显式入参（上游是模型的 `continuing_subword_prefix` 属性）；把词表传 `list[str]` 并用下标当 id（上游是 `dict[str, int]`）；非法输入统一 `ValueError`（上游对参数类型基本不校验，`WordPiece(vocab)` 里 `vocab` 直接进 Rust）；只按空白切词——BertPreTokenizer 的标点切分、`BertNormalizer` 的小写化/去音符/中文插空格都不在本题内。
- `max_input_chars_per_word` 的默认值 100、`##` 的默认续接前缀取自上面的离线源码与实测；`lowercase`、`clean_text`、`handle_chinese_chars` 等归一化开关属于 `BertNormalizer`，本题完全不涉及。
