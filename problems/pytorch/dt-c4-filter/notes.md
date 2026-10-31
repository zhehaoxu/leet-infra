# notes — dt-c4-filter（order 664）

## 回源

- 权威源：`huggingface/datasets` v1.2.1 `datasets/c4/c4_utils.py`（即 TFDS c4 的镜像；HF legacy-datasets/c4 同源）。常量：`_MIN_WORDS_PER_LINE=5`、`_MIN_NUM_SENTENCES=3`、`_MAX_WORD_LENGTH=1000`、`_END_MARKS=(".","?","!","\"")`、`_ELLIPSIS="..."`、`_POLICY_SUBSTRINGS` 6 条。行过滤顺序、整页拒 vs 丢行的区分、badwords 正则 `[^a-z]({})[^a-z]` 全部原样转写。
- 上游 `num_sentences` 用 NLTK punkt 逐行句数；判题环境无 NLTK，题面钉死「每有效行 1 句」作确定性代理（notes 记录偏离）。
- 上游行去重是 Beam 全局 cogroup；本题改为「单页内 + `seen_hashes` 显式跨页状态」，键为 `md5(line.strip().lower())`（上游 `_hash_line` 原样）。

## 判题面

- solve 入参：text、badwords（None = 不过滤）、seen_hashes、三个阈值。返回 accepted/reject_reason/valid_lines/clean_text/num_sentences/drop_counters/new_hashes。
- oracle：`test_cases._py` 独立重写（分步 replace 代替一条 regex），全字段对拍。
- 锚点：六类丢行各一的计数账；引用剥除先于 endmark/词数检查；四类整页拒（lorem/花括号/badword/句数不足）；词长 1000/1001 与词数 4/5 边界；badword 锚点 quirk（essex 不命中、' sex ' 命中）；页内去重与跨页 seen 状态；去重键 strip+lower（大小写/空白变体同键）。
- 非法输入：text 类型、badwords 空串、seen_hashes 类型、阈值 ≤0/bool/浮点 → ValueError。

## 易错点（出题自踩）

- 测试构造行几乎全被 `_MIN_WORDS_PER_LINE=5` 先拦（3–4 词的行永远到不了 javascript/policy 检查）——构造用例必须先数词。
- 整页拒语义钉死：valid_lines 清空、num_sentences 保留拒绝前值、计数器保留。
- `hash-no-strip` 变异等价（line 在循环头已 strip）；真变异是 `hash-case-sensitive`。

## 变异记录（9 条全抓）

ellipsis-kept、longword-ge（边界）、minwords-le（边界）、javascript-rejects（丢行变整页拒）、lorem-startswith、badword-no-anchor（锚点正则）、hash-case-sensitive、seen-not-updated、policy-first（丢行变整页拒）。

## 红线

NLTK/langdetect/Beam 不出；URL 归一化与 RealNews/WET 解析不出（与 `dt-gopher-rules` 划界：那题是 Gopher 九规则，本题是 C4 行过滤 + 去重）。
