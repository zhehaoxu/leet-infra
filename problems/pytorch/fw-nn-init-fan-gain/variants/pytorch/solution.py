import math

import torch


def solve(noises, probe=None):
    """手写 torch.nn.init 的 fan / gain / std / bound 记账（噪声显式入参范式，
    不得调用 torch.nn.init 的任何函数）。

    参数
    ----
    noises : dict[str, Tensor]，键与下方场景一一对应：
      uniform 场景给 U(-1, 1) 样本（rand*2-1），normal 场景给标准正态样本。
    probe : None 或 [nonlinearity, param]——非 None 时只返回
      {"probe_gain": calculate_gain(*probe)}，用于单测 calculate_gain 的
      校验路径（bool param、未知 nonlinearity 等）。

    官方口径（torch/nn/init.py，v2.14.1）：

      _calculate_fan_in_and_fan_out: dim<2 抛 ValueError；
        fan_in = size(1) * prod(shape[2:])，fan_out = size(0) * prod(shape[2:])
      _calculate_correct_fan: mode.lower() 后须在 {"fan_in", "fan_out"}
      calculate_gain:
        linear/conv{1,2,3}d/conv_transpose{1,2,3}d/sigmoid -> 1
        tanh -> 5/3；relu -> sqrt(2)；selu -> 3/4
        leaky_relu -> sqrt(2 / (1 + slope^2))，slope 缺省 0.01；
          param 须为 int(非 bool) 或 float，否则 ValueError
        其他 -> ValueError
      kaiming_uniform_: fan = mode 选定侧；gain = calculate_gain(nonlinearity, a)
        （a 是负斜率，缺省 0——注意不是 calculate_gain 自己的 0.01 缺省）；
        std = gain / sqrt(fan)；bound = sqrt(3) * std；值 = bound * noise
      kaiming_normal_: 同上 std；值 = noise * std
      xavier_uniform_: std = gain * sqrt(2 / (fan_in + fan_out))；
        bound = sqrt(3) * std；值 = bound * noise
      xavier_normal_: std 同上；值 = noise * std
      uniform_(a, b): 值 = a + (b - a) * (noise + 1) / 2
      normal_(mean, std): 值 = mean + std * noise（std < 0 判题面校验 ValueError，
        官方运行时为 RuntimeError，此处统一转写）

    返回
    ----
    dict：
      gains     dict[str, float]：各 nonlinearity 的 gain
      fans      dict[str, [fan_in, fan_out]]
      scenarios dict[str, {"dist": "uniform"|"normal", "param": float,
                           "values": Tensor}]，
                param 对 uniform 场景 = bound、normal 场景 = std
    """
    if not isinstance(noises, dict):
        raise ValueError("noises must be a dict")

    linear_fns = ["linear", "conv1d", "conv2d", "conv3d",
                  "conv_transpose1d", "conv_transpose2d", "conv_transpose3d"]

    def calculate_gain(nonlinearity, param=None):
        if nonlinearity in linear_fns or nonlinearity == "sigmoid":
            return 1
        if nonlinearity == "tanh":
            return 5.0 / 3
        if nonlinearity == "relu":
            return math.sqrt(2.0)
        if nonlinearity == "leaky_relu":
            if param is None:
                negative_slope = 0.01
            elif (not isinstance(param, bool) and isinstance(param, int)) \
                    or isinstance(param, float):
                negative_slope = param
            else:
                raise ValueError(f"negative_slope {param} not a valid number")
            return math.sqrt(2.0 / (1 + negative_slope**2))
        if nonlinearity == "selu":
            return 3.0 / 4
        raise ValueError(f"Unsupported nonlinearity {nonlinearity}")

    if probe is not None:
        return {"probe_gain": calculate_gain(probe[0], probe[1])}

    def fans_of(shape):
        if len(shape) < 2:
            raise ValueError(
                "Fan in and fan out can not be computed for tensor "
                "with fewer than 2 dimensions")
        rf = 1
        for s in shape[2:]:
            rf *= s
        return shape[1] * rf, shape[0] * rf

    def correct_fan(shape, mode):
        mode = mode.lower()
        if mode not in ("fan_in", "fan_out"):
            raise ValueError(f"Mode {mode} not supported, "
                             "please use one of ['fan_in', 'fan_out']")
        fi, fo = fans_of(shape)
        return fi if mode == "fan_in" else fo

    def take(name, shape):
        if name not in noises:
            raise ValueError(f"missing noise: {name}")
        t = noises[name]
        if list(t.shape) != list(shape):
            raise ValueError(f"noise {name} shape mismatch")
        return t

    scenarios = {}

    def add_unif(name, shape, bound, noises_key):
        bound = float(bound)
        u = take(noises_key, shape)
        scenarios[name] = {"dist": "uniform", "param": bound,
                           "values": bound * u}

    def add_norm(name, shape, std, noises_key):
        std = float(std)
        n = take(noises_key, shape)
        scenarios[name] = {"dist": "normal", "param": std,
                           "values": std * n}

    # 1. kaiming_uniform relu fan_in，conv 权重（mode 传大写，官方 lower 后合法）
    shape = [64, 3, 5, 5]
    fan = correct_fan(shape, "FAN_IN")
    gain = calculate_gain("relu")
    add_unif("kaiming_u_conv_relu_fanin", shape,
             math.sqrt(3.0) * gain / math.sqrt(fan),
             "kaiming_u_conv_relu_fanin")
    # 2. kaiming_uniform leaky slope=0.2 fan_out
    shape = [32, 16, 3, 3]
    fan = correct_fan(shape, "fan_out")
    gain = calculate_gain("leaky_relu", 0.2)
    add_unif("kaiming_u_conv_leaky_fanout", shape,
             math.sqrt(3.0) * gain / math.sqrt(fan),
             "kaiming_u_conv_leaky_fanout")
    # 3. kaiming_uniform 全默认（a=0 -> leaky_relu slope 0；fan_in）
    shape = [10, 4]
    fan = correct_fan(shape, "fan_in")
    gain = calculate_gain("leaky_relu", 0)
    add_unif("kaiming_u_lin_default", shape,
             math.sqrt(3.0) * gain / math.sqrt(fan),
             "kaiming_u_lin_default")
    # 4. kaiming_normal relu fan_out
    shape = [8, 6, 3, 3]
    fan = correct_fan(shape, "fan_out")
    gain = calculate_gain("relu")
    add_norm("kaiming_n_conv_relu_fanout", shape, gain / math.sqrt(fan),
             "kaiming_n_conv_relu_fanout")
    # 5. xavier_uniform gain=5/3（tanh）
    shape = [6, 4]
    fi, fo = fans_of(shape)
    std = (5.0 / 3) * math.sqrt(2.0 / float(fi + fo))
    add_unif("xavier_u_tanh", shape, math.sqrt(3.0) * std, "xavier_u_tanh")
    # 6. xavier_normal 全默认 conv
    shape = [4, 2, 2, 2]
    fi, fo = fans_of(shape)
    add_norm("xavier_n_conv", shape,
             math.sqrt(2.0 / float(fi + fo)), "xavier_n_conv")
    # 7. uniform_(a, b) 1D（无 fan；param 统一钉为对称半宽 (b-a)/2）
    shape = [5]
    a, b = -1.5, 2.0
    u = take("plain_uniform_1d", shape)
    scenarios["plain_uniform_1d"] = {
        "dist": "uniform", "param": (b - a) / 2,
        "values": a + (b - a) * (u + 1) / 2}
    # 8. normal_(mean, std)
    shape = [3, 3]
    n = take("plain_normal", shape)
    scenarios["plain_normal"] = {"dist": "normal", "param": 2.0,
                                 "values": 1.0 + 2.0 * n}

    gains = {
        "linear": calculate_gain("linear"),
        "conv2d": calculate_gain("conv2d"),
        "conv_transpose3d": calculate_gain("conv_transpose3d"),
        "sigmoid": calculate_gain("sigmoid"),
        "tanh": calculate_gain("tanh"),
        "relu": calculate_gain("relu"),
        "leaky_relu_default": calculate_gain("leaky_relu"),
        "leaky_relu_02": calculate_gain("leaky_relu", 0.2),
        "selu": calculate_gain("selu"),
    }
    fans = {
        "linear_3x5": list(fans_of([3, 5])),
        "conv_4x3x2x2": list(fans_of([4, 3, 2, 2])),
        "conv1d_2x4x3": list(fans_of([2, 4, 3])),
    }
    return {"gains": gains, "fans": fans, "scenarios": scenarios}
