_ANSWER = "final answer:"
_ACTION = "action:"
_INPUT = "action input:"


def _parse(block):
    """逐行解析一轮：返回 (action, action_input, final_answer)，缺失项为 None。"""
    action = None
    action_input = None
    for raw in block.splitlines():
        text = raw.strip()
        low = text.lower()
        if low.startswith(_ANSWER):
            # 答案一旦出现，本轮后面的一切都不再关心
            return None, None, text[len(_ANSWER):].strip()
        if low.startswith(_INPUT):
            if action_input is None:
                action_input = text[len(_INPUT):].strip()
        elif low.startswith(_ACTION):
            if action is None:
                action = text[len(_ACTION):].strip()
    if action is not None:
        if not action:
            raise ValueError("Action 行为空")
        if action_input is None:
            raise ValueError(f"Action '{action}' 缺少 Action Input")
    return action, action_input, None


def solve(transcripts, tools, max_steps=None):
    if not isinstance(transcripts, list) or not isinstance(tools, dict):
        raise ValueError("transcripts 必须是 list、tools 必须是 dict")
    if max_steps is not None and (not isinstance(max_steps, int) or max_steps < 0):
        raise ValueError("max_steps 必须是非负整数或 None")

    observations = []
    steps = 0
    answer = None
    last = None  # 上一轮真正执行过的 (工具名, 参数)

    for block in transcripts:
        if max_steps is not None and steps >= max_steps:
            break
        if not isinstance(block, str):
            raise ValueError("transcripts 的每一项都必须是 str")
        steps += 1
        action, action_input, final = _parse(block)
        if final is not None:
            answer = final
            break
        if action is None:
            break  # 无 Action：模型自己收手
        if (action, action_input) == last:
            break  # 连续两次同一动作：判定死循环，本次不执行
        last = (action, action_input)
        fn = tools.get(action)
        if fn is None:
            observations.append(f"Error: unknown tool '{action}'")
        else:
            observations.append(str(fn(action_input)))

    return observations, steps, answer
