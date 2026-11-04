"""QUITO 查询引导注意力压缩 — 过滤规则锚点。注意力打分离线入参化。"""
import numpy as np

EXPECTED = {"F1": {"tokens": ["ĠThe", "Ġquick", "Ġbrown", "Ġfox", "Ġjump", "s", "Ġover", "Ġthe", "Ġlazy", "Ġdog", "Ġtoday", "Ġand", "Ġplays", "Ġwell"], "attn_tok": [0.081638, 0.299933, 0.249173, 0.4257, 0.940231, 0.532862, 0.349238, 0.526536, 0.899181, 0.366532, 0.807973, 0.176229, 0.894533, 0.246944], "words": ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog", "today", "and", "plays", "well"], "attn_w": [0.081638, 0.299933, 0.249173, 0.4257, 0.940231, 0.349238, 0.526536, 0.899181, 0.366532, 0.807973, 0.176229, 0.894533, 0.246944], "cnt_w": [1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1]}, "F2": {"tw": ["a", "b", "c", "d", "e", "f"], "ts": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4], "tc": [1, 1, 1, 1, 1, 1], "keep": [True, True, False, False, False, False], "tc2": [1, 2, 1, 1, 1, 1], "keep2": [True, False, False, False, False, False], "ts3": [0.5, 0.9, 0.9, 0.1, 0.9, 0.1], "keep3": [True, True, False, False, False, False]}, "F3": {"sw": ["The", "fox", "runs.", "It", "is", "fast.", "Birds", "fly", "high."], "sa": [0.2, 0.9, 0.3, 0.5, 0.1, 0.8, 0.7, 0.6, 0.4], "sizes": [3, 3, 3], "out": ["The fox runs."], "bp": ["It", "is", "fast."], "bl": 1.5, "out2": ["a", "c"], "out3": ["q", "r"], "bp3": ["p"]}, "F4": {"bw": ["w0", "w1", "w2", "w3", "w4"], "ba": [0.1, 0.3, 0.9, 0.2, 0.4], "sel_w": ["w1", "w2", "w3"], "sel_w2": ["w0", "w1", "w2", "w3", "w4"], "sel_w3": ["x0", "x1"], "sel_w4": ["y0", "y1"]}, "F5": {"s5": [-2.721989, -0.922689, -1.720808, -0.845137, -1.123567, -0.723051, -1.315811, -0.746912, -0.166707, 1.591969, -0.084822, -1.329586, -0.005881], "sigma": 1.5, "smooth": [-1.8985900656843302, -1.6762169353774306, -1.3982768997206205, -1.1833071168092062, -1.04812383331257, -0.951993056082809, -0.7934886888116203, -0.4812472179739002, -0.1011266351176648, 0.08707588498138252, -0.033534835056299554, -0.25948697029215384, -0.37667463074277807], "sigma07": 0.7, "smooth07": [-2.3253073362336942, -1.4727710915441556, -1.3808977867201586, -1.0818979631141323, -0.9916719291356698, -0.928499232994228, -1.064106588563628, -0.7218459947216345, 0.06498441143523968, 0.835461811330902, 0.0038537490202935963, -0.7611204403165076, -0.2911726084426274]}, "F6": {"tw6": ["a", "b", "c", "d", "e", "f"], "ts6": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4], "tc6": [1, 1, 1, 1, 1, 3], "keep6": [True, True, True, False, False, False], "rw7": ["m", "n"], "rs7": [0.5, 0.9], "sw8": ["s0", "s1", "s2"], "sa8": [0.3, 0.9, 0.2], "out8": ["s0", "s1"], "bp8": ["s2"], "bl8": 0.0, "w6b": ["abcd", "ef"], "a6b": [0.6, 0.2], "c6b": [2, 1]}}

CHECKS = [
    "check_reconstruct",
    "check_text_filter",
    "check_sentence_filter",
    "check_dynamic_span",
    "check_gaussian_smooth",
]


def _l(x):
    return np.asarray(x).tolist()


def check_reconstruct(solve):
    f = EXPECTED["F1"]
    got_w, got_a, got_c = solve(mode="reconstruct",
                                tokens=list(f["tokens"]),
                                attention=list(f["attn_tok"]))
    assert got_w == list(f["words"]), got_w
    assert np.allclose(_l(got_a), f["attn_w"], atol=1e-12), got_a
    assert _l(got_c) == list(f["cnt_w"]), got_c
    # 官方怪癖: 首个无前缀 token 走 else 分支 cnt 自增（cnt 偏大 +1）
    w2, a2, c2 = solve(mode="reconstruct",
                       tokens=["hel", "lo", "Ġworld"], attention=[0.1, 0.5, 0.9])
    assert w2 == ["hello", "world"] and _l(c2) == [3, 1] and np.allclose(_l(a2), [0.5, 0.9])
    # Ċ→换行替换先于前缀判断: '\nĠnew' 不以 Ġ 开头 → 整段一词
    w3, _, _ = solve(mode="reconstruct", tokens=["ĊĠnew"], attention=[0.3])
    assert w3 == ["\nĠnew"], w3
    # ▁ 前缀（SentencePiece）与 Ġ 等价开新词
    w6, a6, c6 = solve(mode="reconstruct",
                       tokens=["▁ab", "cd", "▁ef"], attention=[0.4, 0.6, 0.2])
    assert w6 == list(EXPECTED["F6"]["w6b"]), w6
    assert np.allclose(_l(a6), EXPECTED["F6"]["a6b"]) and _l(c6) == list(EXPECTED["F6"]["c6b"])
    # 续词 max 聚合可观测: 子词分低于首子词时取 max
    w4, a4, c4 = solve(mode="reconstruct",
                       tokens=["Ġhi", "gh"], attention=[0.3, 0.7])
    assert w4 == ["high"] and np.allclose(_l(a4), [0.7]) and _l(c4) == [2]
    w5, a5, _ = solve(mode="reconstruct",
                      tokens=["Ġhi", "gh"], attention=[0.9, 0.2])
    assert np.allclose(_l(a5), [0.9])


def check_text_filter(solve):
    f = EXPECTED["F2"]
    rw, rs, keep = solve(mode="text_filter", words=list(f["tw"]),
                         score=list(f["ts"]), cnt=list(f["tc"]), ratio=0.5)
    assert rw == ["a", "b"], rw
    assert np.allclose(_l(rs), [f["ts"][0], f["ts"][1]])
    assert _l(keep) == list(f["keep"])
    # 严格 <: 累积恰好等于预算的词被丢弃并停止
    rw2, _, keep2 = solve(mode="text_filter", words=list(f["tw"]),
                          score=list(f["ts"]), cnt=list(f["tc2"]), ratio=0.4)
    assert rw2 == ["a"], rw2
    assert _l(keep2) == list(f["keep2"])
    # 并列分数 stable: 同分按原序
    rw3, _, _ = solve(mode="text_filter", words=list(f["tw"]),
                      score=list(f["ts3"]), cnt=[1] * 6, ratio=0.5)
    assert rw3 == ["b", "c"], rw3
    # cnt 不同的预算消耗: 大 cnt 词先耗尽预算
    rw4, _, _ = solve(mode="text_filter", words=["u", "v", "w"],
                      score=[0.5, 0.4, 0.3], cnt=[1, 4, 1], ratio=0.5)
    # threshold=3.0: acc1 keep u; acc5 break -> 只留 u
    assert rw4 == ["u"], rw4
    # len(cnt)≠sum(cnt) 阈值口径: sum=8, len=6, ratio .5 → 4.0（len 口径 3.0 会少留一个）
    f6 = EXPECTED["F6"]
    rw6, _, keep6 = solve(mode="text_filter", words=list(f6["tw6"]),
                          score=list(f6["ts6"]), cnt=list(f6["tc6"]), ratio=0.5)
    assert rw6 == ["a", "b", "c"], rw6
    assert _l(keep6) == list(f6["keep6"])
    # 输出保原序（原序 ≠ 分数序）: 分数序是 n,m，原序输出 m,n
    rw7, rs7, _ = solve(mode="text_filter", words=["m", "n", "p"],
                        score=[0.5, 0.9, 0.2], cnt=[1, 1, 1], ratio=0.7)
    assert rw7 == list(f6["rw7"]), rw7
    assert np.allclose(_l(rs7), f6["rs7"]), rs7


def check_sentence_filter(solve):
    f = EXPECTED["F3"]
    f6 = EXPECTED["F6"]
    out, bp, bl = solve(mode="sentence_filter", words=list(f["sw"]),
                        attention=list(f["sa"]), sizes=list(f["sizes"]), ratio=0.5)
    assert out == list(f["out"]), out
    assert bp == list(f["bp"]), bp
    assert abs(_l(bl) - f["bl"]) < 1e-12, bl
    # 等额=预算可过（> 才断，与 text_filter 的严格 < 相反）
    out2, _, _ = solve(mode="sentence_filter", words=["a", "b", "c"],
                       attention=[0.9, 0.3, 0.8], sizes=[1, 1, 1], ratio=2.0 / 3)
    assert out2 == list(f["out2"]), out2
    # 句分并列 stable + 等额可过
    out3, bp3, _ = solve(mode="sentence_filter", words=["p", "q", "r", "s"],
                         attention=[0.5, 0.9, 0.9, 0.2], sizes=[1, 1, 1, 1], ratio=0.5)
    assert out3 == list(f["out3"]), out3
    assert bp3 == list(f["bp3"]), bp3
    # 全部放得下 -> breakpoint None（官方 except 分支钉为 None）
    out4, bp4, bl4 = solve(mode="sentence_filter", words=["x", "y"],
                           attention=[0.8, 0.1], sizes=[1, 1], ratio=1.0)
    assert out4 == ["x", "y"] and bp4 is None and bl4 is None
    # 句分 = 句内 max（非 mean）: 低分长句 vs 高分短句
    out5, _, _ = solve(mode="sentence_filter",
                       words=["aa", "bb", "cc", "dd"],
                       attention=[0.9, 0.1, 0.4, 0.39], sizes=[2, 2], ratio=0.5)
    # budget=2: 句1分0.9选入; 句2分0.4, 2+2>2 断
    assert out5 == ["aa bb"], out5
    # 选中句保原序（原序 ≠ 分数序）: 分数序 s1,s0 → 排序后输出 s0,s1
    out8, bp8, bl8 = solve(mode="sentence_filter", words=list(f6["sw8"]),
                           attention=list(f6["sa8"]), sizes=[1, 1, 1], ratio=2.0 / 3)
    assert out8 == list(f6["out8"]), out8
    assert bp8 == list(f6["bp8"]) and abs(_l(bl8) - f6["bl8"]) < 1e-12, (bp8, bl8)


def check_dynamic_span(solve):
    f = EXPECTED["F4"]
    sw, sa = solve(mode="dynamic_span", bp_words=list(f["bw"]),
                   bp_attn=list(f["ba"]), m=3)
    assert sw == list(f["sel_w"]), sw
    assert np.allclose(_l(sa), [f["ba"][1], f["ba"][2], f["ba"][3]])
    sw2, _ = solve(mode="dynamic_span", bp_words=list(f["bw"]),
                   bp_attn=list(f["ba"]), m=4)
    assert sw2 == list(f["sel_w2"]), sw2
    # m 超句长 break
    sw3, _ = solve(mode="dynamic_span", bp_words=["x0", "x1"],
                   bp_attn=[0.2, 0.8], m=5)
    assert sw3 == list(f["sel_w3"]), sw3
    # 中心在端点: 只向右扩展
    sw4, _ = solve(mode="dynamic_span", bp_words=["y0", "y1", "y2"],
                   bp_attn=[0.9, 0.1, 0.1], m=2)
    assert sw4 == list(f["sel_w4"]), sw4
    # 并列最高分取最左（max 首见）
    sw5, _ = solve(mode="dynamic_span", bp_words=["a", "b", "c"],
                   bp_attn=[0.5, 0.5, 0.1], m=1)
    assert sw5 == ["a"], sw5


def check_gaussian_smooth(solve):
    f = EXPECTED["F5"]
    got = solve(mode="gaussian_smooth", scores=list(f["s5"]), sigma=f["sigma"])
    assert np.allclose(_l(got), f["smooth"], atol=1e-9), np.abs(_l(got) - np.array(f["smooth"])).max()
    # σ=0.7: radius = int(4*0.7+0.5)=3（漏 +0.5 会得 2，可观测差异）
    got07 = solve(mode="gaussian_smooth", scores=list(f["s5"]), sigma=f["sigma07"])
    assert np.allclose(_l(got07), f["smooth07"], atol=1e-9), np.abs(_l(got07) - np.array(f["smooth07"])).max()
    # sigma=0 恒等
    g0 = solve(mode="gaussian_smooth", scores=[1.0, 2.0, 3.0], sigma=0)
    assert np.allclose(_l(g0), [1.0, 2.0, 3.0])
    # 常数序列平滑后仍为常数（reflect 边界）
    gc = solve(mode="gaussian_smooth", scores=[2.0] * 9, sigma=1.5)
    assert np.allclose(_l(gc), 2.0)
    # 输出长度不变
    assert len(_l(got)) == len(f["s5"])
