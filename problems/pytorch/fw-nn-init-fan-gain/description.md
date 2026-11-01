# nn.init 的 fan/gain/std/bound 记账

## 题意

`torch.nn.init` 的六类初始化（Kaiming uniform/normal、Xavier uniform/normal、`uniform_`、`normal_`）全部收敛到一条公式链：**fan 计算 → gain 查表 → std/bound → 噪声缩放**。本题手写整条链（噪声显式入参，不调 `torch.nn.init`），给定固定噪声包 `noises`，输出 gain 表、fan 表与八个场景的 `dist/param/values` 账。

## 求解

官方口径（`torch/nn/init.py` v2.14.1）：

- **fan**：`fan_in = size(1)·prod(shape[2:])`、`fan_out = size(0)·prod(shape[2:])`；`dim<2` 抛 `ValueError`。`mode.lower()` 后须在 `{fan_in, fan_out}`（大写合法）；
- **gain**：linear/conv{1,2,3}d/conv_transpose{1,2,3}d/sigmoid → 1；tanh → 5/3；relu → √2；selu → 3/4；leaky_relu → √(2/(1+slope²))，slope 缺省 **0.01**；param 须为非 bool 的 int 或 float（bool 是 int 子类，官方显式拒）——其余 `ValueError`；
- **kaiming**：`gain = calculate_gain(nonlinearity, a)`——注意 `a` 是负斜率、**缺省 0**（与 calculate_gain 自己的 0.01 缺省不同）；`std = gain/√fan`；uniform 的 `bound = √3·std`；
- **xavier**：`std = gain·√(2/(fan_in+fan_out))`；uniform 的 `bound = √3·std`（= `gain·√(6/(fi+fo))`，两条路径容差内等价）；
- **值变换**（噪声显式入参）：kaiming/xavier uniform 值 = `bound·noise`，normal 值 = `std·noise`；`uniform_(a,b)` 值 = `a + (b-a)·(noise+1)/2`；`normal_(mean,std)` 值 = `mean + std·noise`。

场景含：conv relu fan_in（mode 传大写 `FAN_IN`）、leaky slope=0.2 fan_out、全默认 kaiming（a=0 → gain=√2）、kaiming_normal relu fan_out、xavier gain=5/3、xavier 全默认、1D `uniform_`（无 fan）、`normal_(1,2)`；probe 通道单测 calculate_gain 的校验路径。

## 数据规模

- 噪声包 8 键与场景一一对应，shape 不符/缺键 → `ValueError`；`noises` 非 dict → `ValueError`。
- probe 为 `[nonlinearity, param]`：未知 nonlinearity、bool/str param → `ValueError`。
