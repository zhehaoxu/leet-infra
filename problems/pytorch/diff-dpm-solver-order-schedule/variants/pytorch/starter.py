def solve(
    num_inference_steps,
    solver_order=2,
    lower_order_final=True,
    euler_at_final=False,
    final_sigmas_type="zero",
    algorithm_type="dpmsolver++",
):
    """DPMSolverMultistep 每一步用几阶更新：返回 (orders, counters)。

    num_inference_steps:N，要求 >= 1；solver_order:允许的阶数 1/2/3。

    （"deis" 在上游被静默改写成 "dpmsolver++"，本题同样接受这一别名。）
    algorithm_type 属于 {"dpmsolver", "sde-dpmsolver"}（噪声预测口径）时，
    final_sigmas_type 不能是 "zero"，否则抛 ValueError。

    第 i 步（0 <= i < N）：
      lower_order_final = (i == N-1) and (euler_at_final or (lower_order_final and N < 15)
                                          or final_sigmas_type == "zero")
      lower_order_second = (i == N-2) and lower_order_final and N < 15

    计数 low = 进入该步时的 lower_order_nums（初始 0）：
      若 solver_order == 1 或 low < 1 或 lower_order_final  -> 1 阶
      否则若 solver_order == 2 或 low < 2 或 lower_order_second -> 2 阶
      否则 -> 3 阶
    退出该步时若 low < solver_order，则 low += 1。

    返回 (orders, counters)：
      orders 为长度 N 的 python int 列表（第 i 步实际使用的阶数）；
      counters 为长度 N+1 的 python int 列表，counters[0] = 0，counters[i+1] 是走完第 i 步后的计数。
    """
    # TODO
    raise NotImplementedError
