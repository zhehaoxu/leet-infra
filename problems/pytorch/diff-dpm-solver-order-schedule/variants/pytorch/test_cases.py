def _oracle(steps, order=2, lower_order_final=True, euler_at_final=False, final="zero", algo="dpmsolver++"):
    if algo == "deis":
        algo = "dpmsolver++"
    if algo in ("dpmsolver", "sde-dpmsolver") and final == "zero":
        raise ValueError("noise algorithms need final_sigmas_type='sigma_min'")

    floors = []
    seconds = []
    for index in range(steps):
        floors.append(
            index == steps - 1 and (euler_at_final or (lower_order_final and steps < 15) or final == "zero")
        )
        seconds.append(index == steps - 2 and lower_order_final and steps < 15)

    orders = []
    counters = [0]
    level = 0
    for index in range(steps):
        if order == 1 or level < 1 or floors[index]:
            used = 1
        elif order == 2 or level < 2 or seconds[index]:
            used = 2
        else:
            used = 3
        orders.append(used)
        level = min(level + 1, order)
        counters.append(level)
    return orders, counters


def _run(solve, **kwargs):
    return solve(**kwargs)


def check_example(solve):
    orders, counters = solve(5, solver_order=3)
    assert orders == [1, 2, 3, 2, 1], orders
    assert counters == [0, 1, 2, 3, 3, 3], counters
    assert all(isinstance(v, int) and not isinstance(v, bool) for v in orders + counters)

    orders, counters = solve(4)
    assert orders == [1, 2, 2, 1], orders
    assert counters == [0, 1, 2, 2, 2], counters

    orders, counters = solve(3, solver_order=1)
    assert orders == [1, 1, 1], orders
    assert counters == [0, 1, 1, 1], counters


check_example.example = True


def check_long_run(solve):
    orders, counters = solve(20, solver_order=3)
    assert orders == [1, 2] + [3] * 17 + [1], orders
    assert counters == [0, 1, 2] + [3] * 18, counters


def check_lower_order_final_switch(solve):
    on, _ = solve(5, solver_order=3, lower_order_final=True)
    off, _ = solve(5, solver_order=3, lower_order_final=False, final_sigmas_type="sigma_min")
    assert on == [1, 2, 3, 2, 1], on
    assert off == [1, 2, 3, 3, 3], off

    # final_sigmas_type="zero" 会单独把最后一步压回一阶
    zero, _ = solve(20, solver_order=3, lower_order_final=False)
    assert zero[-1] == 1 and zero[1:-1] == [2] + [3] * 17, zero
    sigma_min, _ = solve(20, solver_order=3, lower_order_final=False, final_sigmas_type="sigma_min")
    assert sigma_min[-1] == 3, sigma_min


def check_fifteen_step_boundary(solve):
    short, _ = solve(14, solver_order=3, final_sigmas_type="sigma_min")
    exact, _ = solve(15, solver_order=3, final_sigmas_type="sigma_min")
    # N<15 时倒数第二步降为二阶、最后一步降为一阶
    assert short[-2:] == [2, 1], short
    assert exact[-2:] == [3, 3], exact


def check_euler_at_final(solve):
    orders, _ = solve(30, solver_order=3, lower_order_final=False, euler_at_final=True, final_sigmas_type="sigma_min")
    assert orders[-1] == 1 and orders[-2] == 3, orders
    plain, _ = solve(30, solver_order=3, lower_order_final=False, final_sigmas_type="sigma_min")
    assert plain[-1] == 3, plain


def check_counters(solve):
    for steps in (1, 2, 3, 7, 14, 15, 20):
        for order in (1, 2, 3):
            orders, counters = solve(steps, solver_order=order)
            assert len(orders) == steps and len(counters) == steps + 1
            assert counters[0] == 0
            assert counters[-1] == min(steps, order), (steps, order, counters)
            for before, after in zip(counters, counters[1:]):
                assert after - before in (0, 1), (steps, order, counters)
            assert all(c <= order for c in counters), (steps, order, counters)
            assert max(orders) <= order


def check_algorithm_alias(solve):
    base, _ = solve(5, solver_order=3)
    alias, _ = solve(5, solver_order=3, algorithm_type="deis")
    assert alias == base
    # 噪声口径的两种算法要求 sigma_min
    for algo in ("dpmsolver", "sde-dpmsolver"):
        noise_orders, _ = solve(5, solver_order=3, algorithm_type=algo, final_sigmas_type="sigma_min")
        assert noise_orders == base, (algo, noise_orders)


def check_against_oracle(solve):
    for steps in range(1, 21):
        for order in (1, 2, 3):
            for flags in (
                dict(),
                dict(lower_order_final=False, final_sigmas_type="sigma_min"),
                dict(euler_at_final=True),
                dict(euler_at_final=True, lower_order_final=False, final_sigmas_type="sigma_min"),
            ):
                want_flags = dict(flags)
                if "final_sigmas_type" in want_flags:
                    want_flags["final"] = want_flags.pop("final_sigmas_type")
                want = _oracle(steps, order=order, **want_flags)
                got = solve(steps, solver_order=order, **flags)
                assert got[0] == want[0], (steps, order, flags, got[0], want[0])
                assert got[1] == want[1], (steps, order, flags, got[1], want[1])


def check_invalid(solve):
    cases = [
        dict(num_inference_steps=0),
        dict(num_inference_steps=-3),
        dict(num_inference_steps=True),
        dict(num_inference_steps=2.0),
        dict(num_inference_steps=5, solver_order=0),
        dict(num_inference_steps=5, solver_order=4),
        dict(num_inference_steps=5, solver_order=True),
        dict(num_inference_steps=5, final_sigmas_type="constant"),
        dict(num_inference_steps=5, algorithm_type="heun"),
        dict(num_inference_steps=5, algorithm_type="dpmsolver"),
        dict(num_inference_steps=5, algorithm_type="sde-dpmsolver"),
        dict(num_inference_steps=5, lower_order_final=1),
        dict(num_inference_steps=5, euler_at_final=None),
    ]
    for kwargs in cases:
        try:
            solve(**kwargs)
            raise AssertionError(f"非法输入应抛 ValueError: {kwargs}")
        except ValueError:
            pass
