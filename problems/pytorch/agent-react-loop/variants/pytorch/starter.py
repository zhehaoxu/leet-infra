def solve(transcripts, tools, max_steps=None):
    """ReAct 循环：解析模型每一轮的文本、执行工具、判断终止。

    transcripts:模型输出列表，每项是一轮的原始文本，形如
        "Thought: ...\nAction: weather\nAction Input: Beijing"
        或 "Thought: ...\nFinal Answer: 22 度"；
    tools:{工具名: callable(str) -> str}；
    max_steps:最多消费几轮，None 表示不限制。

    解析规则（逐行、自上而下）：
    1) 遇到 "Final Answer:" 立即结束，本行冒号后的内容（strip 后）作为答案；
    2) 遇到 "Action:" 取工具名，再找 "Action Input:" 取参数（各取首次出现）；
       有 Action 无 Action Input 抛 ValueError；
    3) 既无 Action 也无 Final Answer → 本轮即空转，终止且无答案。

    执行规则：
    - 工具存在 → 观测 = str(tools[name](action_input))；
    - 工具不存在 → 观测 = "Error: unknown tool '<name>'"（不中断，继续下一轮）；
    - 与上一轮「已执行」的 (工具名, 参数) 完全相同 → 判定死循环，本轮不再执行直接终止。

    返回 (observations, steps, final_answer)：观测列表、已消费轮数、答案（无则 None）。
    """
    # TODO
    raise NotImplementedError
