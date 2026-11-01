import random

import torch


SEEDS_VALID = [0, -5, -2 ** 63, 2 ** 63 - 1, 2 ** 63, 2 ** 64 - 1, 12345]
SEEDS_INVALID = [2 ** 64, 2 ** 65 + 7, -2 ** 63 - 1, True, False, 1.5, "7"]
SEEDS = SEEDS_VALID + SEEDS_INVALID
FRESH_DEFAULT = 67280421310721


def _build_mid(n=5):
    g = torch.Generator().manual_seed(713)
    if n:
        torch.rand(n, generator=g)
    return g


def make_gens(n_mid=5):
    """判题面构造的显式 generator 包（solve 与 oracle 各自独立构造，
    同参数同状态；不触碰全局默认流状态——default 键引用
    torch.default_generator 本身，solve/各自 oracle 调用时全局态一致）。"""
    return {
        "a": torch.Generator().manual_seed(100),
        "b": torch.Generator().manual_seed(100),
        "mid": _build_mid(n_mid),
        "f32": torch.Generator().manual_seed(88),
        "f64": torch.Generator().manual_seed(88),
        "chain": torch.Generator().manual_seed(3),
        "default": torch.default_generator,
    }


def _py(gens, seeds):
    """oracle：直接调官方 Generator.manual_seed/initial_seed（最硬锚点）。
    不调用 torch.manual_seed 等全局函数——避免污染 default 流状态，
    使 oracle 与 solution 看到同一 default 态；官方 fork_rng 语义的
    交叉验证放在各 check 函数里做。"""
    out = {"initial_seed": {}, "seed_errors": {}}
    for s in seeds:
        try:
            g = torch.Generator()
            g.manual_seed(s)
            out["initial_seed"][str(s)] = g.initial_seed()
        except ValueError:
            out["seed_errors"][str(s)] = "ValueError"
        except RuntimeError:
            out["seed_errors"][str(s)] = "RuntimeError"
    out["fresh_default_seed"] = torch.Generator().initial_seed()
    out["same_seed_seqs"] = [
        torch.rand(4, generator=gens["a"]).tolist(),
        torch.rand(4, generator=gens["b"]).tolist(),
    ]
    m = gens["mid"]
    snap = m.get_state()
    first = torch.rand(3, generator=m)
    m.set_state(snap)
    restored = torch.rand(3, generator=m)
    out["midstream"] = {"first_draw": first.tolist(),
                        "restored_draw": restored.tolist(),
                        "equal": bool(torch.equal(first, restored))}
    d = gens["default"]
    snap_d = d.get_state()
    pre = torch.rand(2, generator=d)
    torch.rand(4, generator=d)
    d.set_state(snap_d)
    post = torch.rand(2, generator=d)
    out["fork_default"] = {"pre": pre.tolist(), "post": post.tolist(),
                           "equal": bool(torch.equal(pre, post))}
    out["dtype_streams"] = {
        "f32": torch.rand(3, generator=gens["f32"],
                          dtype=torch.float32).tolist(),
        "f64": torch.rand(3, generator=gens["f64"],
                          dtype=torch.float64).tolist(),
    }
    c = gens["chain"]
    out["consumption_chain"] = {
        "u": torch.rand(2, generator=c).tolist(),
        "n": torch.randn(2, generator=c).tolist(),
        "ints": torch.randint(0, 10, (2,), generator=c).tolist(),
        "perm": torch.randperm(6, generator=c).tolist(),
    }
    return out


def _compare(solve, gens, seeds):
    # fork 账 restore 后 post 消费会把 default 前移；两次调用须看到
    # 同一 default 态 -> solve 前快照、solve 后恢复再跑 oracle
    st = torch.default_generator.get_state().clone()
    led = solve(gens, list(seeds))
    torch.default_generator.set_state(st)
    ref = _py(make_gens(), seeds)
    torch.default_generator.set_state(st)
    assert set(led) == set(ref) == {
        "initial_seed", "seed_errors", "fresh_default_seed",
        "same_seed_seqs", "midstream", "fork_default",
        "dtype_streams", "consumption_chain"}, led.keys()
    for k, v in ref["initial_seed"].items():
        assert led["initial_seed"].get(k) == v, k
    for k, v in ref["seed_errors"].items():
        assert led["seed_errors"].get(k) == v, k
    assert led["fresh_default_seed"] == ref["fresh_default_seed"]
    for i, (a, b) in enumerate(zip(led["same_seed_seqs"],
                                   ref["same_seed_seqs"])):
        assert a == b, i
    assert led["midstream"] == ref["midstream"]
    assert led["fork_default"] == ref["fork_default"]
    assert led["dtype_streams"] == ref["dtype_streams"]
    assert led["consumption_chain"] == ref["consumption_chain"]
    return led


def check_seed_wrap(solve):
    """seed 包装：无符号 mod 2**64（非 int64 重解释），越界/类型分类。"""
    led = _compare(solve, make_gens(), SEEDS)
    isd = led["initial_seed"]
    assert isd["0"] == 0
    assert isd["-5"] == 2 ** 64 - 5          # 负数 -> 无符号大正数
    assert isd[str(-2 ** 63)] == 2 ** 63     # 下边界恰合法
    assert isd[str(2 ** 63)] == 2 ** 63      # 非 int64 负数重解释
    assert isd[str(2 ** 64 - 1)] == 2 ** 64 - 1
    assert isd["12345"] == 12345
    err = led["seed_errors"]
    assert err[str(2 ** 64)] == "ValueError"
    assert err[str(-2 ** 63 - 1)] == "ValueError"
    assert err["True"] == "RuntimeError"     # bool 是 int 子类仍拒
    assert err["False"] == "RuntimeError"
    assert err["1.5"] == "RuntimeError"
    assert err["7"] == "RuntimeError"        # str seed：str('7') == '7'


def check_fresh_default(solve):
    """fresh Generator 缺省种子 = MT19937 黄金常数 67280421310721。"""
    led = _compare(solve, make_gens(), [0])
    assert led["fresh_default_seed"] == FRESH_DEFAULT
    assert torch.Generator().initial_seed() == FRESH_DEFAULT
    # 两个 fresh Generator 序列相同
    assert torch.equal(torch.rand(3, generator=torch.Generator()),
                       torch.rand(3, generator=torch.Generator()))


def check_state_semantics(solve):
    """快照/恢复/fork 语义账。"""
    led = _compare(solve, make_gens(), [0])
    assert led["midstream"]["equal"] is True
    assert led["midstream"]["first_draw"] == led["midstream"]["restored_draw"]
    assert led["fork_default"]["equal"] is True
    assert led["fork_default"]["pre"] == led["fork_default"]["post"]
    # state 是 uint8 张量
    st = torch.Generator().manual_seed(1).get_state()
    assert st.dtype == torch.uint8
    # 同 seed 不同对象序列相同
    a, b = led["same_seed_seqs"]
    assert a == b and len(a) == 4
    # 官方 fork_rng 交叉验证
    torch.manual_seed(21)
    x = torch.rand(3).tolist()
    torch.manual_seed(21)
    with torch.random.fork_rng(devices=[]):
        torch.rand(5)
    assert torch.rand(3).tolist() == x


def check_consumption_routing(solve):
    """dtype 与消费原语进入消耗序列：顺序敏感、同 seed 异 dtype 异流。"""
    led = _compare(solve, make_gens(), [0])
    ds = led["dtype_streams"]
    assert ds["f32"] != ds["f64"]            # dtype 改变消耗路径
    ch = led["consumption_chain"]
    assert len(ch["u"]) == 2 and len(ch["n"]) == 2
    assert len(ch["ints"]) == 2 and sorted(ch["perm"]) == list(range(6))
    # 同 seed rand vs randn 不同
    g = torch.Generator().manual_seed(3)
    u = torch.rand(2, generator=g)
    g = torch.Generator().manual_seed(3)
    n = torch.randn(2, generator=g)
    assert u.tolist() != n.tolist()
    # initial_seed 不推进状态
    g = torch.Generator().manual_seed(42)
    s1 = g.initial_seed()
    x1 = torch.rand(1, generator=g)
    assert g.initial_seed() == s1
    g2 = torch.Generator().manual_seed(42)
    assert torch.equal(x1, torch.rand(1, generator=g2))


def check_randomized(solve):
    """随机对拍：随机合法 seed / 随机中流消耗量，账逐项全等。"""
    rng = random.Random(713)
    for _ in range(60):
        s = rng.randint(-2 ** 63, 2 ** 64 - 1)
        n_mid = rng.randint(0, 9)
        st = torch.default_generator.get_state().clone()
        led = solve(make_gens(n_mid), [s, 0])
        torch.default_generator.set_state(st)
        ref = _py(make_gens(n_mid), [s, 0])
        torch.default_generator.set_state(st)
        assert led["initial_seed"][str(s)] == ref["initial_seed"][str(s)], s
        assert led["same_seed_seqs"] == ref["same_seed_seqs"]
        assert led["midstream"] == ref["midstream"]
        assert led["fork_default"] == ref["fork_default"]
        assert led["consumption_chain"] == ref["consumption_chain"]


def check_default_untouched(solve):
    """fork 账恢复精确性：solve 后 default 态 == 进入前态经过
    rand(3) + [rand(2)+rand(4) 后恢复快照] + rand(2) 的精确结果。"""
    gens = make_gens()
    torch.manual_seed(555)
    st_before = torch.default_generator.get_state().clone()
    torch.rand(3)                          # 进入前消耗
    led = solve(gens, SEEDS)
    probe = torch.Generator()
    probe.set_state(st_before)
    torch.rand(3, generator=probe)         # 进入前消耗
    torch.rand(2, generator=probe)         # fork: pre
    torch.rand(4, generator=probe)         # fork: 块内
    probe.set_state(st_before)
    torch.rand(3, generator=probe)         # 恢复快照（含进入前消耗）
    post2 = torch.rand(2, generator=probe)  # fork: post
    assert post2.tolist() == led["fork_default"]["post"]
    assert torch.equal(torch.default_generator.get_state(),
                       probe.get_state())


def check_invalid_inputs(solve):
    gens = make_gens()
    bad = [
        lambda: solve("not a dict", SEEDS),
        lambda: solve(gens, "not a list"),
        lambda: solve({"a": gens["a"]}, SEEDS),      # 缺 generator 键
    ]
    for fn in bad:
        try:
            fn()
        except (ValueError, KeyError, TypeError):
            pass
        else:
            raise AssertionError("expected error")
    # 分类不抛错：float/bool/str seed 进 seed_errors 账
    led = solve(gens, [3.5, True])
    assert led["seed_errors"] == {"3.5": "RuntimeError",
                                  "True": "RuntimeError"}


def check_no_forbidden_calls(solve):
    """红线：不得出现官方 RNG 高层 API 的真实调用（AST 调用节点级检查，
    docstring/注释字样不算——selftest harness 的模块无 __file__ 时跳过，
    变异 harness 与 integrity 护栏兜底）。"""
    import ast
    import os
    import sys
    mod = sys.modules.get(__name__)
    if mod is None or not getattr(mod, "__file__", None):
        return
    path = os.path.join(os.path.dirname(mod.__file__), "solution.py")
    tree = ast.parse(open(path).read())
    banned_attrs = {"manual_seed", "initial_seed", "fork_rng"}
    banned_roots = {"torch.random"}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names]
            mods = ([node.module] if getattr(node, "module", None) else [])
            if any(n.startswith("torch.random") for n in names + mods):
                raise AssertionError("banned import: torch.random")
        if isinstance(node, ast.Attribute):
            if node.attr in banned_attrs:
                raise AssertionError(f"banned method: .{node.attr}")
            root = node.value
            if isinstance(root, ast.Attribute) and \
                    isinstance(root.value, ast.Name) and \
                    root.value.id == "torch" and root.attr == "random":
                raise AssertionError("banned: torch.random.*")
            if isinstance(root, ast.Name) and root.id == "torch" \
                    and node.attr in ("seed", "manual_seed", "initial_seed"):
                raise AssertionError(f"banned: torch.{node.attr}")
