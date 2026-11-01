# IF-Eval 可验证指令：结构判别器账

## 题意

IF-Eval（google-research/instruction-following-eval，论文 2311.07911）用一组**可程序验证的指令**评估 LLM 的指令遵循。前作（order 665）已收关键词/词数/结尾短语/bullet 六类；本题转写其余三类**结构判别器**的官方 `check_following` 口径：给定 `instruction_ids + kwargs_list + response`，逐条输出 `observed` 与 `pass`，并汇总 `all_pass`。

## 求解

三类指令（与官方实现逐字对齐，纯 stdlib 可复现）：

- `detectable_format:json_format`：`strip()` 后依次 `removeprefix("```json")`、`removeprefix("```Json")`、`removeprefix("```JSON")`、`removeprefix("```")`（各只剥一次），再 `removesuffix("```")`（只剥一次）再 `strip()`；`json.loads` 成功（捕获 `ValueError`）则 pass。注意 JSON 标量 `"42"/"null"/"true"` 合法、Python 大写 `"True"` 非法，而 `"NaN"/"Infinity"` 是 Python json 扩展、合法；
- `detectable_content:postscript`：marker 先 `strip()`；`value.lower()` 后按 marker **精确分支**——`"P.P.S"` → `\s*p\.\s?p\.\s?s.*$`、`"P.S."` → `\s*p\.\s?s\..*$`、其他 → `\s*<marker.lower()>.*$`；`re.findall(..., flags=re.MULTILINE)`，observed = 匹配条数（任意行命中即可）。两个官方 quirk 必须保留：分支比较用原始 marker 大小写（小写 `"p.s."` 走 else、`p\.s\.` 无 `\s?`）；else 分支 marker 拼接进正则**不做 `re.escape`**，marker 里的 `.` 是通配符（marker `"P.S"` 能命中 `pxs hello`）；
- `combination:two_responses`：`split("******")`；空段位于 index 0 或末段则跳过、出现在中间直接 fail；pass = 非空段恰 2 段且 `strip` 后互不相等（比较**不** lower）。observed = 非空段总数。

## 数据规模

- 1 ≤ len(instruction_ids) ≤ 8，指令 id 必须属于 3 类白名单。
- 长度不齐、未知 id、kwargs 条目非 dict、postscript 缺 marker / 非 str / strip 后空、response 非 str → `ValueError`。
