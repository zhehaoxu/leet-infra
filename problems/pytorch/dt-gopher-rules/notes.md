# 解题说明 · Gopher 规则质量过滤

## 上游回源与确定性化

回源 `huggingface/datatrove` 的 `src/datatrove/pipeline/filters/gopher_quality_filter.py`
（Gopher 论文 arXiv:2112.11446 的工程化版本）。三处必须偏离上游才能判题，全部钉进题面：

1. **分词**：上游 `split_into_words` 用 NLTK Treebank word tokenizer（"don't" 会拆成
   "do" + "n't"）；判题环境零依赖，钉死 `text.split()`。
2. **标点集**：上游 `PUNCTUATION_SET` 是 ASCII 标点 + 一大串 unicode 终端标点（约 400 字符）；
   本题钉死 63 字符集（ASCII 全集 + `…—–·、，。！？；：""''《》【】（）〈〉「」『』`）。
3. **规则恒开**：上游每个阈值参数可传 `None` 关闭对应规则；本题恒开，账本固定九条。

## 口径的四个易错点

- **两套词数**：文档长度（规则 1/2）用**非标点词**数；symbol ratio（规则 4/5）的分母是
  **全部词**数。混用会同时破坏规则 1 和规则 4 的账。
- **avg 的字符口径**：非标点词的长度**含词内标点字符**（"don't" 算 5——如果它是一个
  split token）。上游对 non_symbol_words 逐词 `len(w)` 求均值，我们保持一致。
- **strip 的方向**：bullet 只 `lstrip()`（行尾的 `-` 不是 bullet），end ellipsis 只
  `rstrip()`（行首的 `...` 不算）。`"..."` 与 `"…"` 两处都要同时支持（规则 5 是字符计数、
  规则 7 是行尾判定）。
- **短路顺序**：九条按题面顺序检查、首个失败即返回 reason；但 `checks` 账本仍然把九条
  全部算出——变异测试里「调换两条规则的先后」必须被 reason 断言抓住。

## 测试与变异

- oracle 独立手写（`_oracle`，无 import solution），50 组随机文本对拍（词表含标点词、
  省略号、bullet 符号、停用词），阈值参数随机扰动。
- 边界锚点：非标点词数恰好 50/60 通过；`max_avg_word_length=10` 时全 10 字母词通过；
  hash ratio 恰好 0.1（60 词中 6 个 `#`）通过、9 个拒。
- 九条规则各自一个专属用例（构造上要小心前面的规则不先短路：hash/ellipsis/bullet 用例
  都需要先垫够 50+ 非标点词）。
- 变异 9 条全抓：short 边界松弛、long 边界收紧、avg 变严格不等、hash 分母换非标点词、
  `…` 不计入、bullet 不 strip、end ellipsis 不 strip、`any`→`all`、stop 判定 `>=`→`>`。
  其中 endell-nostrip 初版测试没覆盖（行尾无空格时 rstrip 无观察面），补了行尾带空格
  的 `...` 用例后抓到。
