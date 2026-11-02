# ACE 技能书：delta 批次应用与提示词渲染账

ACE 的做法是把「上下文」本身当成一份会进化的策略文档：Generator 干活，Reflector 复盘出洞见，
Curator 把洞见折成一批**增量操作**（加一条、改一条、打个「有用 / 有害」标记、删一条）应用到
技能书上，下一轮再把整本书渲染进提示词。`kayba-ai/agentic-context-engine` 的 `skillbook.py`
就是这本书的实现，本题把它搬成纯函数账。

## 接口

```python
def solve(skills, operations) -> dict
```

`skills` 是初始条目（按插入顺序），每条
`{"id", "section", "keywords", "issue", "insight", "active", "used", "helpful", "harmful", "neutral"}`；
`operations` 是批次，每条形如
`{"type": "ADD"|"UPDATE"|"TAG"|"REMOVE"|"MARK_USED", "section", "issue", "keywords",
"insight", "skill_id", "delta", "purge", "skill_ids"}`。

返回 `order`、`skills`（id → 条目，含已软删的）、`sections`（节 → id 列表，允许重复）、
`active`、`prompt`（渲染出的提示词）、`stats`、`log`（每条操作的结果）。

## 输入输出

```
输入：skills=[]
     operations=[{type:"ADD", section:"Tooling", issue:"prefer ripgrep",
                  keywords:["Search"], skill_id:"l1"}]
输出：skills["l1"].section   = "context"          # 非 context/harness 折叠
     skills["l1"].keywords  = ["tooling","search"]
     skills["l1"].insight   = "prefer ripgrep"    # 用 issue 兜底
     sections              = {"context": ["l1"]}
```

## 规则

- **文本归一化**：所有文本先 `str()` 再 `strip()`，空串归一成 `None`；`issue`、`section` 必须
  非空，否则 `ValueError`。
- **关键词归一化**：`strip` → `lower` → `\s+` 变 `_`，空串丢弃，**按首次出现去重**。
- **section 折叠**：只有 `context` / `harness` 合法（大小写不敏感）；其它值一律折叠成
  `context`，同时把**原 section 名**（已归一化）插到关键词队首，且 `insight` 缺省时用 `issue`
  兜底。真正的 `context` 条目缺 `insight` 要报错，`harness` 不要求。
- **ADD**：关键词为空时默认 `[section]`；同 `skill_id` 重复 ADD 会覆盖记录，但节里的 id 列表
  **再登记一次**（于是渲染时会出现两遍）。不传 `skill_id` 时按 `"<section>-<序号>"` 生成。
- **UPDATE**：`issue` / `insight` 传 `None` 表示「不改」；`keywords` 传**空列表**也表示「不改」，
  但传非空列表而里面全是空白则报错；改完 `context` 条目必须仍有 `insight`。`skill_id` 缺失或
  指向不存在的条目 → 记 `noop`，不报错。
- **TAG**：`delta` 只按**符号**取 ±1（`7` 与 `1` 一样只加一次），`0` 记 `neutral`。
- **REMOVE**：默认软删（`active=False`，记录仍在）；`purge=True` 才真删，并把 id 从节列表里摘掉，
  节空了连键一起删。
- **MARK_USED**：只给 `active` 的条目加 `used`，返回值/日志里只列真正加上的 id。
- **渲染**：固定 `context` → `harness` 顺序；每节先输出 `## <section>`，每条输出
  `- [<id>]`、`  Keywords: a, b`、`  Issue: ...`，**只有 `insight` 非空**才追加 `  Insight: ...`；
  每条后面跟一个空行，最后整串 `rstrip()`。空书渲染成空串。
- **stats**：`sections` 是「有 active 条目的节数」（不是节键数），`skills` 是**含软删**的总数，
  `active_skills` 只数 active，`by_section` 恒含两个节键且从 0 起。
- **校验**：条目 id 唯一、`section` 合法、计数为非负整数（`bool` 拒收）；未知操作类型、
  `delta` 非整数、`skill_ids` 非序列一律 `ValueError`。

## 提示

- 「空列表 = 不改」与「空串 = 清空」是两条**不同**的哨兵语义：`keywords: []` 保留原关键词，
  而 `insight: ""` 会把 harness 条目的 insight 真的清成 `None`。
- 软删的条目仍占 `stats["skills"]` 的分子，但不进 `active`、不进渲染、不被 `MARK_USED` 计数。
- 同一 id 重复 ADD 会让节列表出现重复 id，于是渲染时同一条印两遍——这是官方实现的行为，
  不是本题引入的怪癖。
