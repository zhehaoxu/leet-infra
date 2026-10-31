# IF-Eval 可验证指令：规则判别器账

## 题意

IF-Eval（google-research/instruction-following-eval，2508.09108 系）用一组**可程序验证的指令**评估模型的指令遵循：关键词存在/禁用/词频、词数约束、结尾短语、bullet 列表数等。本题转写 6 类判别器的 strict 口径：给定 `instruction_ids + kwargs_list + response`，逐条输出 `observed`（观测计数）与 `pass`，并汇总 `all_pass`。

## 求解

口径题面钉死（NLTK 的确定性代理全部换成显式规则）：

- `keywords:existence`：子串、大小写不敏感，全部命中才 pass；
- `keywords:forbidden`：同上子串口径，命中数必须为 0；
- `keywords:frequency`：`r"\b<kw>\b"` 在 lower 后的匹配数（词边界防止 `spa` 命中 `spam`），按 `at least`/`less than` 比较；
- `length_constraints:number_words`：`len(response.split())` 空白切分（标点附着不切词）；
- `format:end_phrase`：`response.strip().rstrip('"').lower()` 后 endswith（尾引号剥除、大小写不敏感）；
- `detectable_format:number_bullet_lists`：以 `* ` 开头的行数 ≥ num_bullets（`- ` 不算）。

## 数据规模

- 1 ≤ len(instruction_ids) ≤ 8，指令 id 必须属于 6 类白名单。
- 长度不齐、未知 id、空 keywords、frequency/num_words ≤0 或 bool、relation 非法、空 end_phrase、response 非 str → `ValueError`。
