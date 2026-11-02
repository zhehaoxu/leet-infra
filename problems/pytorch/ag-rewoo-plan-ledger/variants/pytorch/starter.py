def solve(planner_response, worker_outputs, solver_output,
          planner_tokens=0, solver_tokens=0, tools=("Google", "LLM"),
          planner_model="text-davinci-003", solver_model="text-davinci-003"):
    """ReWOO（2305.18323）官方 billxbf/ReWOO 的 algos/PWS.py 标量化账。

    Planner 输出文本 / Worker 输出（按证据号）/ Solver 输出全部显式入参，
    只考三段账：

    1. Plan 解析：以 "Plan:" 开头的行进 plans；"#E<d> = ..." 行按第一个
       '=' 切分（两侧 strip），证据号 strip 后**恰为 "#Ed"（单数字下标，
       len==3）才有效**，否则该行记 "No evidence found"（多位下标 #E12
       的工具调用被整体丢弃）。
    2. Worker 替换（按证据插入序逐条）：无 '[' 的声明行原文即证据
       （pass-through）；有 '[' 的行 split("[",1) 取工具名，其余部分
       **盲删最后一个字符**（约定为 ']'），再按原文出现序对 #E\\d+ 占位符
       replace 全量替换为 "[证据]"——**插入文本中新出现的占位符也会被
       后续替换命中**；尚未算出的前向引用原样保留。工具在 tools 中才执行
       （证据 = worker_outputs[e]），否则证据 = "No evidence found"；
       Google 计调用次数，LLM 计 len(调用串+证据)//4 个 token。
    3. worker_log：第 i 个 Plan 行与证据 "#E{i+1}" 按**下标**配对拼接
       "plan\\nEvidence:\\n证据\\n"；成本账：token_cost = planner 价 ×
       planner_tokens + solver 价 × solver_tokens + 固定 davinci-003 价 ×
       LLM token；tool_cost = Google 次数 × 0.01；steps = len(plans)+1。

    价格表：text-davinci-003=0.00002、gpt-3.5-turbo=0.000002、gpt-4=0.00003、
    tloen/alpaca-lora-7b 与 rewoo/planner_7B=0.0，未知名 ValueError
    （"Model not found"）。

    返回账本 dict（plans/planner_evidences/worker_calls/worker_evidences/
    worker_log/output/tool_usage/steps/total_tokens/token_cost/tool_cost/
    total_cost）。域约束：response/solver_output 为字符串、worker_outputs
    为字典、tools 为字符串列表、token 数为非负 int；证据行缺 '='、有效
    工具调用缺 worker 输出、Plan 行缺配对证据 → ValueError。
    """
    raise NotImplementedError
