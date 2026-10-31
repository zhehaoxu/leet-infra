
def solve(think_steps, budget, final_tokens, wait_steps=(), wait_cost=0):
    """按题面钉死的三段状态机执行 budget forcing 账本。

    规则：
      1. 思考步按序发射：emitted + c <= budget 整步发射；否则发射剩余
         budget-emitted 个、forced_final=True、思考终止（length forcing）。
      2. 未被截断时按序尝试 wait_steps：每次要求 remaining >= wait_cost，
         允许则扣 wait_cost 再按规则 1 发射该 Wait 步（可能再次截断，
         截断后停止后续 Wait）；不允许则停止尝试。已截断则跳过全部 Wait。
      3. final_tokens 按 remaining 截断发射；被切则 answer_truncated=True。
    """
    raise NotImplementedError
