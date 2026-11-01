import math

import torch
import torch.nn.init as tinit


SCEN_SHAPES = {
    "kaiming_u_conv_relu_fanin": [64, 3, 5, 5],
    "kaiming_u_conv_leaky_fanout": [32, 16, 3, 3],
    "kaiming_u_lin_default": [10, 4],
    "kaiming_n_conv_relu_fanout": [8, 6, 3, 3],
    "xavier_u_tanh": [6, 4],
    "xavier_n_conv": [4, 2, 2, 2],
    "plain_uniform_1d": [5],
    "plain_normal": [3, 3],
}
UNIF_SCEN = {"kaiming_u_conv_relu_fanin", "kaiming_u_conv_leaky_fanout",
             "kaiming_u_lin_default", "xavier_u_tanh", "plain_uniform_1d"}


def make_noises():
    """固定 seed 710 的显式噪声包（float64 保证 1e-12 级对拍容差）：
    uniform 场景 U(-1,1)，normal 场景标准正态。"""
    g = torch.Generator().manual_seed(710)
    noises = {}
    for name, shape in SCEN_SHAPES.items():
        if name in UNIF_SCEN:
            noises[name] = torch.rand(shape, generator=g, dtype=torch.float64) * 2 - 1
        else:
            noises[name] = torch.randn(shape, generator=g, dtype=torch.float64)
    return noises


def _py(noises, probe=None):
    """oracle：直接调 torch.nn.init 官方 gain/fan 函数（最硬锚点），
    值变换按官方 docstring 公式（bound = gain*sqrt(3/fan)、
    xavier bound = gain*sqrt(6/(fi+fo))）独立成链。"""
    if probe is not None:
        return {"probe_gain": tinit.calculate_gain(probe[0], probe[1])}
    out = {"gains": {
        "linear": tinit.calculate_gain("linear"),
        "conv2d": tinit.calculate_gain("conv2d"),
        "conv_transpose3d": tinit.calculate_gain("conv_transpose3d"),
        "sigmoid": tinit.calculate_gain("sigmoid"),
        "tanh": tinit.calculate_gain("tanh"),
        "relu": tinit.calculate_gain("relu"),
        "leaky_relu_default": tinit.calculate_gain("leaky_relu"),
        "leaky_relu_02": tinit.calculate_gain("leaky_relu", 0.2),
        "selu": tinit.calculate_gain("selu"),
    }, "fans": {}, "scenarios": {}}
    for name, shape in [("linear_3x5", [3, 5]), ("conv_4x3x2x2", [4, 3, 2, 2]),
                        ("conv1d_2x4x3", [2, 4, 3])]:
        t = torch.empty(shape)
        fi, fo = tinit._calculate_fan_in_and_fan_out(t)
        out["fans"][name] = [int(fi), int(fo)]

    def bound_kaiming(shape, mode, nonlinearity, a):
        fan = tinit._calculate_correct_fan(torch.empty(shape), mode)
        return tinit.calculate_gain(nonlinearity, a) * math.sqrt(3.0 / fan)

    def xavier_pair(shape, gain):
        fi, fo = tinit._calculate_fan_in_and_fan_out(torch.empty(shape))
        std = gain * math.sqrt(2.0 / (fi + fo))
        return gain * math.sqrt(6.0 / (fi + fo)), std

    specs = [
        ("kaiming_u_conv_relu_fanin", "uniform",
         bound_kaiming([64, 3, 5, 5], "fan_in", "relu", 0)),
        ("kaiming_u_conv_leaky_fanout", "uniform",
         bound_kaiming([32, 16, 3, 3], "fan_out", "leaky_relu", 0.2)),
        ("kaiming_u_lin_default", "uniform",
         bound_kaiming([10, 4], "fan_in", "leaky_relu", 0)),
        ("kaiming_n_conv_relu_fanout", "normal",
         tinit.calculate_gain("relu")
         / math.sqrt(tinit._calculate_correct_fan(
             torch.empty([8, 6, 3, 3]), "fan_out"))),
    ]
    for name, dist, param in specs:
        n = noises[name]
        out["scenarios"][name] = {"dist": dist, "param": param,
                                  "values": param * n}
    b, std = xavier_pair([6, 4], 5.0 / 3)
    out["scenarios"]["xavier_u_tanh"] = {
        "dist": "uniform", "param": b, "values": b * noises["xavier_u_tanh"]}
    _, std = xavier_pair([4, 2, 2, 2], 1.0)
    out["scenarios"]["xavier_n_conv"] = {
        "dist": "normal", "param": std, "values": std * noises["xavier_n_conv"]}
    a, b2 = -1.5, 2.0
    u = noises["plain_uniform_1d"]
    out["scenarios"]["plain_uniform_1d"] = {
        "dist": "uniform", "param": (b2 - a) / 2,
        "values": 0.5 * (u + 1) * (b2 - a) + a}
    n = noises["plain_normal"]
    out["scenarios"]["plain_normal"] = {
        "dist": "normal", "param": 2.0, "values": 1.0 + 2.0 * n}
    return out


def _compare(solve, noises, probe=None):
    led = solve(dict(noises), probe)
    ref = _py(noises, probe)
    if probe is not None:
        assert math.isclose(led["probe_gain"], ref["probe_gain"],
                            rel_tol=1e-12, abs_tol=1e-15), (led, ref)
        return led
    assert set(led) == set(ref) == {"gains", "fans", "scenarios"}, led.keys()
    for k in ref["gains"]:
        assert math.isclose(led["gains"][k], ref["gains"][k],
                            rel_tol=1e-12, abs_tol=1e-15), k
    for k in ref["fans"]:
        assert led["fans"][k] == ref["fans"][k], k
    for k, sc in ref["scenarios"].items():
        got = led["scenarios"][k]
        assert got["dist"] == sc["dist"], k
        assert math.isclose(got["param"], sc["param"],
                            rel_tol=1e-12, abs_tol=1e-15), k
        assert torch.allclose(got["values"], sc["values"],
                              rtol=1e-12, atol=1e-15), k
    return led


def check_gains(solve):
    led = _compare(solve, make_noises())
    assert math.isclose(led["gains"]["relu"], math.sqrt(2), rel_tol=1e-12)
    assert math.isclose(led["gains"]["tanh"], 5.0 / 3, rel_tol=1e-12)
    assert math.isclose(led["gains"]["selu"], 0.75, rel_tol=1e-12)
    assert led["gains"]["linear"] == 1 and led["gains"]["sigmoid"] == 1
    assert math.isclose(led["gains"]["leaky_relu_default"],
                        math.sqrt(2.0 / 1.0001), rel_tol=1e-12)


def check_fans(solve):
    led = _compare(solve, make_noises())
    assert led["fans"]["linear_3x5"] == [5, 3]
    assert led["fans"]["conv_4x3x2x2"] == [3 * 4, 4 * 4]  # (12, 16)
    assert led["fans"]["conv1d_2x4x3"] == [12, 6]


def check_scenarios(solve):
    led = _compare(solve, make_noises())
    sc = led["scenarios"]
    # kaiming relu fan_in：bound = sqrt(3)*sqrt(2)/sqrt(75) = sqrt(0.08)
    assert math.isclose(sc["kaiming_u_conv_relu_fanin"]["param"],
                        math.sqrt(0.08), rel_tol=1e-12)
    assert sc["kaiming_u_conv_relu_fanin"]["dist"] == "uniform"
    # leaky slope 0.2 fan_out(288)：bound = sqrt(3/288)*sqrt(2/1.04)
    assert math.isclose(sc["kaiming_u_conv_leaky_fanout"]["param"],
                        math.sqrt(3.0 / 288) * math.sqrt(2.0 / 1.04),
                        rel_tol=1e-12)
    # 全默认 kaiming：a=0 -> gain=sqrt(2)（不是 calculate_gain 的 0.01 缺省）
    assert math.isclose(sc["kaiming_u_lin_default"]["param"],
                        math.sqrt(6.0 / 4), rel_tol=1e-12)
    # kaiming_normal relu fan_out(72)
    assert math.isclose(sc["kaiming_n_conv_relu_fanout"]["param"],
                        math.sqrt(2.0 / 72), rel_tol=1e-12)
    # xavier_uniform gain=5/3, fi+fo=10
    assert math.isclose(sc["xavier_u_tanh"]["param"],
                        math.sqrt(3.0) * (5 / 3) * math.sqrt(0.2),
                        rel_tol=1e-12)
    assert math.isclose(sc["xavier_n_conv"]["param"], math.sqrt(2.0 / 24),
                        rel_tol=1e-12)  # fi=8, fo=16
    # plain uniform/normal
    assert math.isclose(sc["plain_uniform_1d"]["param"], 1.75, rel_tol=1e-12)
    assert math.isclose(sc["plain_normal"]["param"], 2.0, rel_tol=1e-12)
    # 值 = 噪声线性缩放：取一个样本核对
    u0 = make_noises()["kaiming_u_conv_relu_fanin"].flatten()[0]
    v0 = sc["kaiming_u_conv_relu_fanin"]["values"].flatten()[0]
    assert math.isclose(float(v0), math.sqrt(0.08) * float(u0), rel_tol=1e-12)


def check_probe(solve):
    g = _compare(solve, make_noises(), ["leaky_relu", 0.2])
    assert math.isclose(g["probe_gain"], math.sqrt(2.0 / 1.04), rel_tol=1e-12)
    g2 = _compare(solve, make_noises(), ["relu", None])
    assert math.isclose(g2["probe_gain"], math.sqrt(2), rel_tol=1e-12)
    # bool param 拒绝（官方 quirk：bool 是 int 子类，须显式拒）
    # int param 合法
    g3 = _compare(solve, make_noises(), ["leaky_relu", 0])
    assert math.isclose(g3["probe_gain"], math.sqrt(2), rel_tol=1e-12)


def check_vs_torch_fill(solve):
    """端到端锚点：solution 的 bound 与官方 kaiming_uniform_ 的采样极值区间一致
    （大样本 min/max 逼近 ±bound），且 fan/gain 与官方函数全等。"""
    noises = make_noises()
    led = _compare(solve, noises)
    bound = led["scenarios"]["kaiming_u_conv_relu_fanin"]["param"]
    big = torch.empty(256, 3, 5, 5)  # fan_in=75，与场景 1 同构
    tinit.kaiming_uniform_(big, mode="fan_in", nonlinearity="relu")
    assert abs(float(big.min()) + bound) < 1e-3
    assert abs(float(big.max()) - bound) < 1e-3
    # 官方 gain/fan 全等
    assert math.isclose(led["gains"]["leaky_relu_02"],
                        tinit.calculate_gain("leaky_relu", 0.2), rel_tol=1e-12)
    assert led["fans"]["conv_4x3x2x2"] == list(
        tinit._calculate_fan_in_and_fan_out(torch.empty([4, 3, 2, 2])))


def check_invalid_inputs(solve):
    noises = make_noises()
    bad = [
        lambda: solve("not a dict"),
        lambda: solve({k: v for k, v in noises.items()
                       if k != "plain_normal"}),   # 缺噪声键
        lambda: solve({**noises, "plain_normal": torch.zeros(4)}),  # shape 错
        lambda: solve(noises, ["gelu", None]),     # 未知 nonlinearity
        lambda: solve(noises, ["leaky_relu", True]),   # bool param
        lambda: solve(noises, ["leaky_relu", "0.2"]),  # str param
    ]
    for fn in bad:
        try:
            fn()
        except (ValueError, IndexError):
            pass
        else:
            raise AssertionError("expected error")
    # 大写 mode 场景 1 仍须成功（官方 lower）
    _compare(solve, noises)
