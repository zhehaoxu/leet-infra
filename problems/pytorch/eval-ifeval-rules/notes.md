# notes — eval-ifeval-rules（order 665）

## 回源

- 出处：`google-research/instruction-following-eval`（`instructions.py` 的 REGISTRY；论文 2508.09108「Instruction-Following Evaluation for Large Language Models」）。上游判别器用 NLTK word_tokenize/sent_tokenize 且有 strict/loose 两口径；判题环境无 NLTK，题面只收 **strict 口径**并把词法代理钉死（词数 = `split()`、频率 = `\b` 正则）。
- 只收 6 类：keywords:existence / forbidden / frequency、length_constraints:number_words、format:end_phrase、detectable_format:number_bullet_lists。上游其余类别（json_format、postscript、capital_word_frequency、combination:two_responses、sentences 计数等）留作后续候选。

## 判题面

- solve 入参：instruction_ids/kwargs_list/response。返回 results（每条 observed + pass）与 all_pass。
- oracle：`test_cases._py` 独立重写（`[1:] == "* "` 代替 startswith、while 剥引号等不同实现路径），随机 120 组对拍。
- 锚点：existence 子串大小写不敏感；forbidden 子串口径（`secrets` 命中 `secret`）；频率 `\b` 词边界（`spa` 不命中 `spam`、行尾标点不挡）；词数边界 5（at least / less than 各测）；end_phrase 尾引号剥除 + 大小写不敏感 + 中途出现不算；bullet `* ` 计数、`- ` 不算；三条指令组合 all_pass。
- 非法输入：空 id 列表、未知 id、kwargs 缺键/空列表、frequency/num_words/num_bullets ≤0 或 bool、relation 非法、空 end_phrase、response 非 str → ValueError。

## 变异记录（9 条全抓）

existence-or（全命中变任一命中）、forbidden-inverted、freq-rel-flip、words-count-chars（字符数代词数）、endphrase-case-sensitive、endphrase-no-quote-strip、bullets-dash-too、existence-case-sensitive、freq-no-boundary（count 代 \b 正则）。

## 红线

模型调用、loose 口径的 markdown 剥离族不出；与 `rs-*`（推理系统）划界：本题是**指令遵循评估**，无候选聚合语义。
