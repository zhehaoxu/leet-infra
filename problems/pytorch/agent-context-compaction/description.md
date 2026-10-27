# Agent 基建：上下文窗口预算压缩

## 问题

agent 跑得越久，上下文越满：system prompt、完整历史、全部工具 schema、scratchpad、中间工具返回，很快就顶到窗口上限。直接截断尾部最省事，但会把 system 约束和工具说明一起丢掉——模型立刻开始乱调工具。所以生产实现都改成「按优先级挑片段」：钉住的先留，剩下的在预算里贪心装填。

（LangChain 的 `trim_messages`、Claude Code 的 auto-compact、OpenHands 的 condenser 都是这套思路的不同旋钮组合。）

## 任务

实现 `solve(segments, budget, pinned_priority=100)` → `(kept, used)`：

- `segments`：`[(name, tokens, priority)]`，按时间顺序给出；`budget`：窗口允许的总 token；`pinned_priority`：优先级 `>=` 该值的片段视为**钉住**（必须保留）。
- 钉住片段先占预算；若它们的 token 总和已超过 `budget` → `ValueError`。
- 其余片段按 `(优先级降序, 原始下标升序)` 贪心尝试：装得下就留，装不下就**跳过**（跳过不是终止，后面的小片段仍要试）。
- `kept` 按**原始时间顺序**升序返回，`used` 是保留片段的 token 总和。

## 示例

```
segments = [
    ("system", 40, 100),
    ("history", 120, 10),
    ("tools", 60, 60),
    ("scratchpad", 30, 40),
]

solve(segments, 200) -> ([0, 2, 3], 130)
```

装填顺序是 system(40) → tools(60) → scratchpad(30) → history(120，装不下被跳过)；最终保留 40+60+30=130 个 token，返回的时间序下标是 `[0, 2, 3]`。

## 约束

- **钉住片段不可压缩**：它们承载 system 约束与工具契约，丢了就等于换了任务；超预算时应当显式报错，而不是静默丢弃。
- **跳过大块、继续尝试小块**：`break` 会让一个超长历史段吃掉后面所有名额，`continue` 才能把预算填满。
- **返回时间序而不是优先级序**：拼进 prompt 时必须保持对话的时间结构，优先级只用于挑选。
- **同优先级按下标先到先得**：否则同样的输入在不同实现里给出不同的保留集合。
- **非法输入抛 `ValueError`**：`segments` 不是 `list`、`budget` 为负、`tokens` 为负或非整数、片段不是三元组。
