# notes — eval-ifeval-format（order 709）

## 回源

- 出处：`google-research/google-research` 子目录 `instruction_following_eval/instructions.py`（2023 原始 commit `e4c89b6404` 与 2026 master 逐行比对，本题三类 checker 无差异）；registry key：`detectable_format:json_format` / `detectable_content:postscript` / `combination:two_responses`。论文 2311.07911。
- **capital_word_frequency 与 sentences 计数不收**：官方从 2023 原始版起就用 `nltk.word_tokenize`（内部 punkt 句切）与 `nltk.data.load("nltk:tokenizers/punkt/english.pickle")`，判题环境无 NLTK 数据包，语义无法精确复现，留待环境具备后再议。order 665 notes 里"`\b[A-Z]+\b` regex 口径"系误记，官方从未用过该实现。

## 判题面

- solve 入参：instruction_ids/kwargs_list/response，返回 results（每条 observed + pass）与 all_pass，接口与 order 665 对齐。
- oracle：`test_cases._py` 独立重写——json 用 `re.sub(r"^```(?:json|Json|JSON)?", ...)` 正则剥 tick（官方 removeprefix 链的等价正则），postscript 用 `splitlines` + 逐行 `re.search`（代替 MULTILINE findall，随机对拍不构造 `\r` 保证等价），two_responses 用列表推导 + `parts[1:-1]` 空段判定；随机 120 组对拍。
- **observed 口径自钉**（官方只返回 bool，observed 是本题补充账面）：json_format = bool；postscript = MULTILINE 匹配条数（= 命中行数）；two_responses = 非空段总数（中间空段场景也计全部非空段——官方提前 return，收集等价于全部非空段）。

## 锚点（手算 + 官方逻辑逐例验证）

- json_format：```json / ```Json / ```JSON / ``` 四个前缀各只剥一次、`removesuffix("```")` 只剥一次；`"42"/"null"/"true"/"NaN"/"Infinity"` 合法、`"True"` 非法（手算修正：Python json 只扩展接受 NaN/Infinity，不接受大写布尔）；单引号/尾逗号/空串/纯 ```/6 反引号只剥一次剩 ```{}```/`"```json{...}```json"` 尾不剥 全部 fail；外围空白 pass；`\xa0`（NBSP）只在**最外层 strip 或最终 strip** 可达处被剥——tick 包裹的 NBSP（` ```json\xa0{}\xa0 `）只能靠最终 strip，去掉它该用例即挂。
- postscript：P.P.S 分支 `\s?p\.\s?p\.\s?s` 两处**各只认一个空白**（两空格 fail）；P.S. 分支 `p\.\s?s\.` 要求 s 后紧跟点；分支比较用原始 marker 大小写——小写 `"p.s."` 走 else（`p\.s\.` 无 `\s?`，`"p. s. x"` fail 而大写 marker pass）；**marker 不 re.escape**：else 拼接后 `.` 是通配（marker `"P.S"` 命中 `pxs hello`；marker `"P.P.S."` 的 `p.p.s..` 两个连续 `.` 吃掉两空格，`"P.P.S  x"` pass）；MULTILINE 任意行命中即算、多行各计一次。
- two_responses：恰两段且 strip 后不等（不 lower，`"Same******same"` pass）；三段 fail；首尾空段跳过（`"******a******"` 只剩 1 段 fail；`"x ****** y ******"` 仍两段 pass）；中间空段直接 fail；5 星不分隔、7 星 split 剩 `["", "*"]`、12 星剩两空段 全 fail。
- 组合语义：两 checker 共享同一 response——postscript 只能藏进 JSON 字符串值里才能 json+postscript 双 pass；`"```json\n{...}\n```\nP.S. ..."` 整体不可解析 → json fail。

## 变异记录（17 条全抓）

json-prefix-Json / json-prefix-JSON（大小写变体）、json-suffix-tick、json-tick-global-strip（replace 全剥 vs 单次剥）、json-container-only（只收 dict/list → 标量挂）、json-catch-narrow（except TypeError 漏捕）、ps-sstar（\s?→\s*）、ps-branch-swap（两分支互换）、ps-no-multiline、ps-no-lower、ps-escape-marker（加 re.escape → 通配 quirk 挂）、ps-pass-ge0、two-lower-eq（比较前 lower）、two-no-strip-eq（比较不 strip）、two-len-ge1、two-inner-empty-ok、two-star5（5 星分隔）。

等价变异 1 条记 notes：json 前缀 `"```Json"` 与 `"```JSON"` 的 removeprefix 顺序互换——两模式大小写互斥，顺序无关（SURVIVED 确认）。`json-final-strip-drop` 初判等价，实为非等价（tick 包裹 NBSP 用例抓）。

## 红线

模型调用、NLTK 依赖族（capital_word_frequency/sentences）、strict/loose 双口径的 markdown 剥离族不出；与 order 665 划界：本题只收结构类（JSON 包裹/落款/双答案），关键词与计数类不重复。
