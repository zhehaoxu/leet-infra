import torch


def solve(gens, seeds):
    """RNG 可复现账（torch 2.14.1 CPU 口径，torch/nn/__init__.py 与
    torch/_C/Generator 语义）。手写 seed 包装、状态快照与消费路由——
    不得调用 torch.manual_seed / torch.seed / torch.initial_seed /
    torch.random 模块的任何函数（含 fork_rng），不得调用
    Generator.manual_seed / Generator.initial_seed；随机消费原语只允许
    torch.rand / torch.randn / torch.randint / torch.randperm 且必须配
    显式 generator=...；状态原语只允许 Generator.get_state / set_state。

    参数
    ----
    gens : dict[str, Generator]，全部由判题面构造并播种（solve 不得改播种）：
      "a"   : manual_seed(100)，未消费
      "b"   : manual_seed(100)，未消费（与 a 同 seed、不同对象）
      "mid" : manual_seed(713) 后判题面已消费 torch.rand(5)（中流状态）
      "f32" : manual_seed(88)
      "f64" : manual_seed(88)
      "chain": manual_seed(3)
      "default": torch.default_generator（全局默认生成器对象本身，
                 播种状态由判题面设定，solve 消费后必须完整恢复）
    seeds : list，待分类的 seed（含合法 int、越界 int、bool/float/str）

    官方口径（v2.14.1 实测钉死）：

      seed 包装：合法域 [-2**63, 2**64-1]；initial_seed 返回
        s mod 2**64 的**无符号**值（-5 -> 2**64-5；2**63 -> 2**63，
        非 int64 重解释的负数）；越界 -> ValueError
      类型拒绝：bool / float / str -> RuntimeError（bool 是 int 子类，
        须先于 int 分支显式拒）
      fresh Generator：未播种的 initial_seed = 67280421310721
        （MT19937 缺省种子常数），且两个 fresh Generator 序列相同
      状态快照：get_state 返回 uint8[5056]；set_state 精确恢复到
        快照时刻（中流恢复后重复消费得到相同值）
      fork 语义：快照 -> 消费 -> 恢复 -> 再消费 == 快照时刻的直接续消费
      同 seed 不同 Generator 对象 -> 序列逐元素相同
      dtype 进入消费路径：同 seed 的 float32 / float64 消耗不同流
      消费原语共享 generator 状态，顺序敏感：
        rand / randn / randint / randperm 依次消耗

    需要产出的账（键固定）：
      initial_seed      {str(seed): 无符号包装值}（仅合法 int）
      seed_errors       {str(seed): "ValueError" | "RuntimeError"}
      fresh_default_seed 未播种 Generator 的 initial_seed 常数
      same_seed_seqs    [rand(4)@a.tolist, rand(4)@b.tolist]
      midstream         {"first_draw", "restored_draw", "equal"}
                        mid：快照 -> rand(3) -> 恢复快照 -> rand(3)
      fork_default      {"pre", "post", "equal"}
                        default：快照 -> rand(2) -> rand(4) -> 恢复快照
                        -> rand(2)（post 须与 pre 逐元素相同）
      dtype_streams     {"f32": rand(3)@f32, "f64": rand(3)@f64}
      consumption_chain chain：rand(2) -> randn(2) -> randint(0,10,(2,))
                        -> randperm(6) 的结果账
    """
    if not isinstance(gens, dict) or not isinstance(seeds, (list, tuple)):
        raise ValueError("bad inputs")

    out = {"initial_seed": {}, "seed_errors": {}}
    for s in seeds:
        # 官方类型拒绝：bool 是 int 子类，须先拒；float/str -> RuntimeError
        if isinstance(s, bool) or not isinstance(s, int):
            out["seed_errors"][str(s)] = "RuntimeError"
            continue
        # 合法域 [-2**63, 2**64-1]，越界 -> ValueError
        if -(2 ** 63) <= s <= 2 ** 64 - 1:
            out["initial_seed"][str(s)] = s % (2 ** 64)
        else:
            out["seed_errors"][str(s)] = "ValueError"
    out["fresh_default_seed"] = 67280421310721

    # 同 seed 不同对象：序列逐元素相同
    out["same_seed_seqs"] = [
        torch.rand(4, generator=gens["a"]).tolist(),
        torch.rand(4, generator=gens["b"]).tolist(),
    ]

    # 中流快照恢复：快照 -> 消费 -> 恢复 -> 再消费 == 恢复时刻续消费
    m = gens["mid"]
    snap = m.get_state()
    first = torch.rand(3, generator=m)
    m.set_state(snap)
    restored = torch.rand(3, generator=m)
    out["midstream"] = {"first_draw": first.tolist(),
                        "restored_draw": restored.tolist(),
                        "equal": bool(torch.equal(first, restored))}

    # fork 语义（手写）：快照 -> 消费 -> 旁路消费 -> 恢复 -> 消费 == pre
    d = gens["default"]
    snap_d = d.get_state()
    pre = torch.rand(2, generator=d)
    torch.rand(4, generator=d)          # fork 块内消耗（不回滚前先消耗）
    d.set_state(snap_d)
    post = torch.rand(2, generator=d)
    out["fork_default"] = {"pre": pre.tolist(), "post": post.tolist(),
                           "equal": bool(torch.equal(pre, post))}

    # dtype 进入消费路径
    out["dtype_streams"] = {
        "f32": torch.rand(3, generator=gens["f32"],
                          dtype=torch.float32).tolist(),
        "f64": torch.rand(3, generator=gens["f64"],
                          dtype=torch.float64).tolist(),
    }

    # 消费链：rand -> randn -> randint -> randperm 顺序敏感
    c = gens["chain"]
    out["consumption_chain"] = {
        "u": torch.rand(2, generator=c).tolist(),
        "n": torch.randn(2, generator=c).tolist(),
        "ints": torch.randint(0, 10, (2,), generator=c).tolist(),
        "perm": torch.randperm(6, generator=c).tolist(),
    }
    return out
