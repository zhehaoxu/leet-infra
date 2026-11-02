def solve(events, num_islands=10, functions_per_prompt=2,
          cluster_sampling_temperature_init=0.1,
          cluster_sampling_temperature_period=30000, function_name="f"):
    # TODO: FunSearch 岛屿程序库的注册 / 取样 / 重置账。
    #   events 三种：
    #     {"kind": "register", "program": {"id","doc","size"}, "scores": {...}, "island": int|None}
    #     {"kind": "prompt", "island": int, "clusters": [...], "programs": [...]}
    #     {"kind": "reset", "founders": [...]}
    #   - 簇键 = 逐测试分数按**测试名升序**拼成的 tuple；簇分 = **最后插入**那个测试的分数，
    #     且只在首个成员进簇时定下，后来者只进程序表。
    #   - island=None 表示种子程序，广播到所有岛，每岛各记一次。
    #   - 岛内最优只在分数**严格**更高时替换（同分留先到）。
    #   - 取样温度 = t_init * (1 - (岛累计程序数 % period) / period)；簇概率是 分数/温度 的 softmax。
    #   - 取 min(簇数, functions_per_prompt) 个簇（可重复），按簇分升序编号 v0..v(n-1)，
    #     v0 保留原 docstring、v(i) 换成 "Improved version of `<f>_v{i-1}`."，最后再加 v(n) 的空壳头。
    #   - 簇内取样概率 = softmax(-(len - min)/(max + 1e-6), 1.0)。
    #   - 重置 `num_islands // 2` 个岛：按 (岛内最优, 岛号) 升序取弱的；founders 显式给出创始人，
    #     创始人必须是保留岛且手上有程序，克隆其最优程序后该岛最优分随之继承。
    raise NotImplementedError
