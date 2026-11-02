# notes（eval-judge-score-parse）

## 回源

- 主源：`lm-sys/FastChat` 的 `fastchat/llm_judge/common.py`（main 分支，实抓核对）
  - 34-37 行四条正则（成对主 / 备份、单点主 / 备份），逐字照抄；`TIE_DELTA = 0.1` 在 30 行。
  - `run_judge_single`(135-189)：`re.search(one_score_pattern, judgment)`，失败再试 `_backup`，
    仍失败 `rating = -1`；成功走 `ast.literal_eval(match.groups()[0])`。
  - `run_judge_pair`(235-310)：291-304 行的成对分支——先主模式后备份，
    `scores = [ast.literal_eval(s.strip()) for s in match.groups()]`，
    `abs(scores[0]-scores[1]) <= TIE_DELTA` → `"tie"`，否则比大小给 `"A"` / `"B"`，抠不到给 `"error"`。
- 单点表源：同仓 `fastchat/llm_judge/show_result.py` 的 `display_result_single`（9-36 行）：
  20 行 `df[df["score"] != -1]` 剔除解析失败；26-36 行按 `(model, turn)` 与 `model` 求均值后降序打印。

## 划界（重要）

- `eval-judge-swap-consistency`（order 684）已经占了**标记式**判决（`[[A]]` / `[[B]]` / `[[C]]`）
  + swap 两局摆位映射 + 不一致即平局 + 半胜聚合。那题完全不碰数值分数串，
  本题也完全不碰 swap 与胜率表——两题的输入形态、解析对象、输出都不同。
- 本题首版曾把 swap 映射与 `win_rate_adjusted` 也做进来，与 684 重合约六成，
  按「撞车/无新机制，勿再立项」的口径**砍掉重写**：只留数值解析 + 单点均值表。
  这也是 todos 里原候选的表述（「single 模式 TIE_DELTA 与 `[[a,b]]` 双分数 regex」）。
- `tie_delta` 在本题只用于**单条文内的 A/B 高下**，不做跨文一致性与场次结算。

## 本题的规范化

- 官方把解析散在两处（`common.py` 抠分、`show_result.py` 求均值），中间夹着 pandas；
  本题合成一个纯函数，judge 文本作显式入参，模型调用完全不出。
- **排序的平分规则必须自己钉**：官方 `df.sort_values(...)` 默认 `quicksort` 不是稳定排序，
  同分顺序由 pandas 内部实现决定、跨版本可能变。本题统一改成「均值降序 + 模型名升序」。
- 解析细节都实测过（`.venv-eval` + CPython 3.13）：
  - `\s?` 恰好一个：`[[8,\t7]]`、`[[8,\n7]]`、`[[8,\xa07]]`、`[[8,\u30007]]` 命中；
    `[[8,  7]]` 主备份全失配（`\s` 只吃一个，剩下的空格让数字位对不上）。
  - `[[8,7],[9,2]]` 主模式整体不中（外层不是 `]]`），备份模式取**最左**的内层 `[8,7]`；
    `[[8,7]] and [[9,2]]` 主模式取最左的完整块；`[[8, 7]]x` 命中、`[[8, 7]x` 退到备份模式。
  - `[[ 8, 7]]` 两条都不中（`[[` 后必须紧跟数字）；`[[-1, 7]]` 也抠不出（数字不带符号）。
  - `5.` → `5.0`（float），`007` → `ast.literal_eval` 抛 **SyntaxError**（官方未捕获、服务端会崩）；
    本题归一化成 `ValueError`，`check_rejections` 兜住。
  - 浮点边界两例：`abs(8.0 - 8.1) = 0.09999999999999964 <= 0.1` → 平；
    `abs(1.0 - 1.1) = 0.10000000000000009 > 0.1` → 不平。反过来
    `abs(5.5 - 5.55) = 0.04999999999999982` 落在 `tie_delta=0.05` **之下**（名义恰等却判平），
    这三条一起把 `<=` 的浮点两侧钉死。
- `s.strip()` 在官方是死代码（捕获组只含 `\d` 与 `.`，不可能带空白），本题保留由 `literal_eval`
  产生的 `int` / `float` 区分，并在 `check_number_kinds` / `check_contract` 里断言。
- `-1` 是「抠不到」的哨兵，而 `0` 是合法分数：本题专门有 `[[0]]` 用例，
  防止用 `sc or -1` 这类写法把 0 吞掉（`m-score-or` 变异即此）。

## 变异

23 条全部 CAUGHT（`.muts-judge2.json`，跑完即删）：

- 解析层：两条 `\s?`→`\s*`（成对主 / 备份各一）、成对备份优先、单点备份优先；
- 数值层：全部转 `float`、`except` 不收 `SyntaxError`；
- 判定层：`<=` → `<`、A/B 方向反转、抠不到改判 `"tie"`；
- 单点表：`-1` 不剔除、`avg` 只取 `turn_1`、`turn_1` / `turn_2` 为空仍强算（除零）、
  分桶忽略 `turn`（全塞进第 1 轮）、`sc or -1` 吞掉 0 分；
- 排序层：升序、去掉名字吊分；
- 校验层：`turn ∈ {0,1,2}`、放行 `turn=True`、放行 `tie_delta=True`、放行负 `tie_delta`、`mode` 校验失效、
  `entries` 只认 `list`（拒 tuple）。

**等价变异（1 条，已实证）**：`sc[0] > sc[1]` → `sc[0] >= sc[1]`。看着像能翻转平局，实则不可观测——
相等的一对必然先被 `abs(a-b) <= tie_delta` 拦下（`tie_delta >= 0` 保证 `<=` 覆盖 0），
比较式永远见不到相等的操作数，故语义完全等价，不记入清单。

首轮另有两条必须记名的教训：① `[[0008]]` 放在成对模式里根本不会进入解析（单数字文本），
前导零用例必须放到 `[[007, 7]]` / `[[8, 08]]` 这种真的能抠出 token 的位置；
② 首版 `m-pair-text-check` 写成了 `text = text` 的空操作，白跑一轮——写变异也要先看清改了什么。

## 判题注意

- 容差：均值用 1e-12 比较（`22/3` 这类值要容纳 1e-16 级误差），`scores` 与 `turn` 保持精确相等。
- `pattern` 标签是判据的一部分（`two` / `two_backup` / `one` / `one_backup` / `none`），
  不能只对分数——否则「主模式还是备份模式命中的」这一层信息全部丢失。
