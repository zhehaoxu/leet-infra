# Agent 基建：ReAct 循环状态机

## 问题

LLM 本身只会吐文本，真正干活的是外面的 agent 运行时：它从模型输出里解析出 `Thought` / `Action` / `Action Input` / `Final Answer`，调用对应工具，把 `Observation` 拼回上下文，再决定继续还是收手。ReAct（Yao et al., 2210.03629）把这条循环写成了固定协议，LangChain 的 `AgentExecutor`、AutoGPT、各家 function-calling 循环都是它的变体。

本题把这层状态机单独拎出来考：**模型每一轮输出了什么是显式入参**，你只负责解析、执行、判终止，不涉及任何生成。

## 任务

实现 `solve(transcripts, tools, max_steps=None)`：

- `transcripts`：模型输出列表，一项是一轮的原始文本；`tools`：`{工具名: callable(str) -> str}`；`max_steps`：最多消费几轮，`None` 不限。
- 解析（逐行、自上而下、各类信息取首次出现）：遇到 `Final Answer:` 立刻结束，冒号后的内容（strip）作答案；遇到 `Action:` 取工具名并继续找 `Action Input:`，有 Action 没有输入抛 `ValueError`；两者都没有 → 本轮空转，终止且无答案。
- 执行：工具存在则观测 = `str(tools[name](action_input))`；工具不存在则观测 = `Error: unknown tool '<name>'` 且**不中断**；与上一轮已执行的 `(工具名, 参数)` 完全相同 → 判定死循环，本轮不执行直接终止。
- 返回 `(observations, steps, final_answer)`：观测列表、已消费轮数、答案（没有则 `None`）。

## 示例

```
transcripts = [
    "Thought: check the weather first\nAction: weather\nAction Input: Beijing",
    "Thought: now the clock\nAction: clock\nAction Input: now",
    "Thought: enough\nFinal Answer: Beijing 22C at 10:00",
]
tools = {"weather": lambda city: "22C", "clock": lambda when: "10:00"}

solve(transcripts, tools) -> (["22C", "10:00"], 3, "Beijing 22C at 10:00")
```

## 约束

- **终止有四种来源**：拿到 `Final Answer`、本轮既无 Action 也无答案、连续重复动作、`max_steps` 用尽或脚本耗尽。
- **`steps` 计的是消费轮数**，包括「产生答案的那一轮」和「触发死循环护栏的那一轮」；`max_steps=0` 时一轮都不消费。
- **未知工具是正常路径**：真实系统里模型会幻觉出不存在的工具，运行时给一条错误观测让它自我纠正，而不是崩掉。
- **空答案 ≠ 没有答案**：`Final Answer:` 后面什么都没有时返回空串 `""`，而不是 `None`。
- **答案优先**：同一轮里既写了 Action 又写了 Final Answer 时，以 `Final Answer` 为准、不执行工具。
