_ALGORITHMS = ("dpmsolver++", "sde-dpmsolver++", "dpmsolver", "sde-dpmsolver")
_NOISE_ALGORITHMS = ("dpmsolver", "sde-dpmsolver")
_FINAL = ("zero", "sigma_min")


def _int(name, value, low, high):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} 必须是整数")
    if value < low or value > high:
        raise ValueError(f"{name} 必须落在 [{low}, {high}]")
    return value


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
    steps = _int("num_inference_steps", num_inference_steps, 1, 10**9)
    order = _int("solver_order", solver_order, 1, 3)
    if not isinstance(lower_order_final, bool):
        raise ValueError("lower_order_final 必须是 bool")
    if not isinstance(euler_at_final, bool):
        raise ValueError("euler_at_final 必须是 bool")
    if final_sigmas_type not in _FINAL:
        raise ValueError(f"未知的 final_sigmas_type: {final_sigmas_type!r}")
    if algorithm_type == "deis":
        algorithm_type = "dpmsolver++"
    if algorithm_type not in _ALGORITHMS:
        raise ValueError(f"未知的 algorithm_type: {algorithm_type!r}")
    if algorithm_type in _NOISE_ALGORITHMS and final_sigmas_type == "zero":
        raise ValueError("dpmsolver / sde-dpmsolver 不支持 final_sigmas_type='zero'")

    orders = []
    counters = [0]
    low = 0
    for index in range(steps):
        floor_final = (index == steps - 1) and (
            euler_at_final or (lower_order_final and steps < 15) or final_sigmas_type == "zero"
        )
        second_final = (index == steps - 2) and lower_order_final and steps < 15

        if order == 1 or low < 1 or floor_final:
            used = 1
        elif order == 2 or low < 2 or second_final:
            used = 2
        else:
            used = 3
        orders.append(used)

        if low < order:
            low += 1
        counters.append(low)

    return orders, counters
