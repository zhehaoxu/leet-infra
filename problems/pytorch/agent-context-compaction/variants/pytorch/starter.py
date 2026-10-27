def solve(segments, budget, pinned_priority=100):
    """在上下文窗口预算内挑选要保留的片段。

    segments:[(name, tokens, priority)]，按时间顺序给出（system / history / tools / scratchpad …）；
    budget:窗口允许的总 token 数；
    pinned_priority:优先级 >= 该值的片段视为钉住（必须保留）。

    规则：
    1) 钉住片段先占预算；若它们的 token 总和已超过 budget → ValueError；
    2) 其余片段按 (优先级降序, 原始下标升序) 贪心尝试：装得下就留、装不下就跳过
       （**跳过不是终止**：后面的小片段仍要试）；
    3) 返回值里的下标按**原始时间顺序**升序排列（不是优先级顺序）。

    返回 (kept, used)：保留片段的原始下标列表、它们占用的 token 总数。
    """
    # TODO
    raise NotImplementedError
