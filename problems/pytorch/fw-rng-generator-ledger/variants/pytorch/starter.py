def solve(gens, seeds):
    """RNG 可复现账：seed 包装、显式 generator 与状态快照。

    手写 seed 分类/包装、状态快照 fork 语义与消费路由——不得调用
    torch.manual_seed / torch.seed / torch.initial_seed / torch.random
    模块（含 fork_rng），不得调用 Generator.manual_seed /
    Generator.initial_seed；消费原语只允许 torch.rand/randn/randint/
    randperm 配显式 generator=...；状态原语只允许 get_state/set_state。

    官方口径（torch 2.14.1 CPU 实测钉死）：
      - seed 合法域 [-2**63, 2**64-1]；initial_seed 返回 s mod 2**64
        的**无符号**值（-5 -> 2**64-5；2**63 -> 2**63，非 int64 重解释）；
        越界 -> ValueError
      - bool/float/str seed -> RuntimeError（bool 是 int 子类，先拒）
      - fresh Generator 未播种 initial_seed = 67280421310721
        （MT19937 缺省种子常数），两个 fresh Generator 序列相同
      - get_state 返回 uint8[5056]；set_state 精确恢复到快照时刻；
        fork 语义：快照 -> 消费 -> 旁路消费 -> 恢复 -> 消费 == pre
      - 同 seed 不同 Generator 对象 -> 序列逐元素相同
      - dtype 进入消费路径（同 seed float32/float64 异流）
      - 消费原语顺序敏感：rand -> randn -> randint -> randperm

    参数 gens 由判题面构造（"a"/"b"/"mid"/"f32"/"f64"/"chain"/
    "default" 七键，default 即 torch.default_generator 本身）。

    返回 dict：initial_seed / seed_errors / fresh_default_seed /
    same_seed_seqs / midstream / fork_default / dtype_streams /
    consumption_chain
    """
    raise NotImplementedError
