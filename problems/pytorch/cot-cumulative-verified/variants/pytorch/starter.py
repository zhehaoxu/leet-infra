"""Cumulative Reasoning（Game24）累积态验证账 -- 你的实现。

官方（iiis-ai/cumulative-reasoning, arXiv:2308.04371）Game of 24 的
solve()（Game24/game24-openai.py L537-590）把 proposer / valid 判定 /
verifier 三类 LLM 调用组织成状态图生长循环。本题把三类 LLM 输出全部
显式入参（rounds 脚本），请实现下面的 solve；不允许 import numpy /
guidance / openai。

参数
----
mode : str
    "solve"（默认）| "state" | "dist"。
puzzle : str
    初始状态串（如 '2, 4, 6'），records 的根。
rounds : list[dict]
    每轮 {chosen, next, remaining, judgement, verify, equation}：
    chosen=本轮采样到的已有状态；next=proposer 算式（原始串）；
    remaining=proposer 剩余数字串（原始串）；judgement=valid 判定
    （'Invalid' 拦截）；verify=verifier 结论（'sure'/'likely'/其他）；
    equation=verifier 附带的合并算式（sure 时取首行）。
trycnt : int
    循环上界（rounds 超出部分被忽略；不足则提前按失败收场）。
index, n_states : int
    mode='dist' 用：当前轮下标与当前状态数。

返回
----
mode="solve"：(success, marker, steps)。
    每轮顺序（官方钉死，任何异常静默跳过该轮）：
      a = remaining.strip().split('\\n')[0].strip()；
      门 A：next.strip() 含 [^0-9+\\-*/.(),=\\s] 之外字符 -> 跳过；
      门 B：next.strip() 或 remaining.strip() 无 \\S -> 跳过；
      judgement == 'Invalid' -> 跳过；
      verify == 'sure'：steps = [equation 首行, next.strip() + ' (left a)',
        沿父链回溯各步（各拼 ' (left 状态)'）] 再整体 reverse；
        返回 (True, 轮下标, steps)；
      verify == 'likely' 且 a 不在 records：forbidden[chosen] 追加
        next.strip()，forbidden[a]=[]，records 追加 a，parent[a]=chosen，
        last_step[a]=next.strip()；a 已在 records 整轮静默丢弃；
      其他 verify 无动作。
    无 success：返回 (False, trycnt, [])。
mode="state"：{"records", "parent", "last_step", "forbidden"} 快照。
mode="dist"：采样分布 list[float]——index < 1/2*trycnt（浮点除）时
    根偏置：n>1 则其余项 0.5/(n-1)、p[0]=0.5，n==1 则 p=[1.0]；否则
    每项 1/n。

异常
----
mode 未注册、puzzle 非 str、trycnt/index/n_states 非 int 或越界、
rounds 非 list、轮缺键 -> ValueError；轮内坏值（chosen 不在 records、
next=None 等）-> 静默跳过该轮（官方 try/except continue 语义）。
"""


def solve(mode="solve", puzzle=None, rounds=None, trycnt=None,
          index=None, n_states=None):
    raise NotImplementedError("TODO: 实现 dist / solve / state 三入口")
