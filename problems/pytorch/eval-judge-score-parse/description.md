# LLM-as-judge：分数串解析与单点评分表

## 题目描述

裁判模型（judge）输出一段自由文本，末行写 `[[8, 7]]`（成对打分）或 `[[8]]`（单点打分）。服务端并不理解文本，只用两条正则把分数抠出来；抠不到就判 `error`，单点模式则记 `-1` 并在统计时丢掉。

本题只做这一段：**分数串的四条正则解析**（主 / 备份两级、`\s?` 的恰一空白、`\d+\.?\d*` 的无符号数字）、`ast.literal_eval` 的 `int` / `float` 区分、成对模式的 `tie_delta` 判定，以及单点模式的 `(model, turn)` 均值表。位置偏差的 swap 两局、不一致即平局与胜率聚合属于另一题（`eval-judge-swap-consistency`），本题不涉及。

## 规则

签名 `solve(entries, mode="pair", tie_delta=0.1)`。

**1. 四条正则**（与 FastChat `llm_judge/common.py` 同源）：

- 成对主 `\[\[(\d+\.?\d*),\s?(\d+\.?\d*)\]\]`、成对备份 `\[(\d+\.?\d*),\s?(\d+\.?\d*)\]`；
- 单点主 `\[\[(\d+\.?\d*)\]\]`、单点备份 `\[(\d+\.?\d*)\]`。

都是「全串搜索、取最左命中」，所以分数可以夹在句子里、尾部有无文字都无所谓，且**主模式失败才试备份模式**。三个细节必须复刻：

- `\s?` 是**至多一个**空白字符，且 `\s` 按 Unicode 判定：空格、制表符、换行、`\xa0`（NBSP）、`\u3000`（全角空格）都算一个；写两个空格则一个都不算，主备份同时失配。
- 数字是 `\d+\.?\d*`：不接受负号（`[[-1, 7]]` 抠不出任何分数）、不接受省掉整数部分的 `.5`、接受尾巴带点的 `5.`。
- 主模式要求 `[[` 后**紧跟**数字，`[[ 8, 7]]` 主备份都不中；`[[8, 7]]x` 仍命中（无尾锚）。

抠出的子串按 Python 字面量解析：无小数点得 `int`、有得 `float`（`5.` → `5.0`）；前导零的 `007` 不是合法字面量，抛 `ValueError`。

**2. 成对模式**（`mode="pair"`，`entries` 是字符串列表）：每条给出 `{"pattern", "scores", "verdict"}`。`pattern` ∈ `{"two", "two_backup", "none"}`；抠不到时 `scores = None`、`verdict = "error"`；否则 `scores` 是两个数字，`|a - b| <= tie_delta` 判 `"tie"`，否则 A 位高判 `"A"`、B 位高判 `"B"`。此模式 `table == {}`、`rank == []`。

**3. 单点模式**（`mode="single"`，`entries` 是 `{"model", "turn", "text"}` 列表）：每条给出 `{"model", "turn", "pattern", "score"}`，`pattern` ∈ `{"one", "one_backup", "none"}`，抠不到时 `score = -1`。

统计时先剔除 `score == -1` 的行，再按模型汇总：`turn_1` / `turn_2` 分别是该轮分数的算术均值（该轮没有有效行则为 `None`），`avg` 是全部有效行的均值。`rank` 按 `avg` 降序、同分按模型名升序。全是 `-1` 的模型不出现在 `table` 里。

## 示例

```text
输入：entries = ["[[8, 7]]", "[[7, 8]]", "[[9, 9]]"], mode = "pair"
输出：rows = [{"pattern": "two", "scores": [8, 7], "verdict": "A"},
              {"pattern": "two", "scores": [7, 8], "verdict": "B"},
              {"pattern": "two", "scores": [9, 9], "verdict": "tie"}]
      table = {}, rank = []
```

```text
输入：entries = [{"model": "a", "turn": 1, "text": "[[10]]"},
                 {"model": "a", "turn": 1, "text": "bad"},
                 {"model": "a", "turn": 2, "text": "[[4]]"}],
      mode = "single"
输出：rows = [{"pattern": "one", "score": 10}, {"pattern": "none", "score": -1},
              {"pattern": "one", "score": 4}]
      table = {"a": {"turn_1": 10.0, "turn_2": 4.0, "avg": 7.0}}
      rank  = ["a"]
```

## 约束

- `mode` 只能是 `"pair"` / `"single"`；`tie_delta` 必须是有限非负实数且拒收 `bool`；`entries` 必须是 list 或 tuple。
- 成对模式每条必须是 `str`；单点模式每条必须是含 `model` / `turn` / `text` 的 `dict`，`model` 非空字符串、`turn` 取 `1` 或 `2` 的 `int`（拒收 `bool` 与 `1.0`）、`text` 是字符串。
- 空 `entries` 合法，返回空表；返回字段固定为 `mode` / `rows` / `table` / `rank`，非当前模式的字段为空。
- `0` 是合法分数（不等于 `-1` 哨兵）；入参不得被就地改写；`scores` 保留 `int` / `float` 的原始区分。
