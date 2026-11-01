# notes — fw-nn-init-fan-gain（order 710）

## 回源

- 出处：本地判题环境 `.venv-eval` 的 `torch/nn/init.py`（v2.14.1，2026-10-19 安装）——比 GitHub 抓取更权威（判题即此版本）；关键函数 `calculate_gain` / `_calculate_fan_in_and_fan_out` / `_calculate_correct_fan` / `kaiming_uniform_` / `kaiming_normal_` / `xavier_uniform_` / `xavier_normal_` 逐行核对。He 初始化论文 1502.01852、Glorot 2010。
- 与 todos 候选的划界：只收 fan/gain/std/bound 公式链账；`trunc_normal_`（erf/CDF 重参数化）、`orthogonal_`（QR）、`sparse_`、`dirac_` 不入题。

## 判题面

- solve(noises, probe=None)：噪声显式入参范式（判题环境无全局 RNG 依赖——torch.Generator 只在测试侧生成固定噪声包）；probe=[nonlinearity, param] 时只返回 `{"probe_gain": ...}`，用于从外部触发 calculate_gain 校验路径（bool param、未知 nonlinearity 在固定场景里不可达）。
- oracle：`test_cases._py` **直调 torch.nn.init 官方函数**（`tinit.calculate_gain` / `tinit._calculate_fan_in_and_fan_out` / `tinit._calculate_correct_fan`）作最硬锚点，值变换按官方 docstring 公式独立成链（bound = gain·√(3/fan)、xavier bound = gain·√(6/(fi+fo))——与 solution 的官方实现路径 √3·std 故意不同，对拍容差吸收 ULP）。`check_vs_torch_fill` 端到端：solution 的 bound 与官方 `kaiming_uniform_` 大样本采样的 min/max 极值区间核对（[256,3,5,5]，fan_in=75）。
- 浮点对拍：噪声包用 float64（float32 ULP ~6e-8，1e-12 容差下必挂）；标量 isclose(rel 1e-12)、values torch.allclose(rtol 1e-12)。

## 锚点

- fan：[3,5]→(5,3)、[4,3,2,2]→(12,16)、[2,4,3]→(12,6)（receptive field = shape[2:] 连乘）。
- gain：bool param 拒绝（官方 `not isinstance(param, bool) and isinstance(param, int) or isinstance(param, float)`，运算优先级 = `(A and B) or C`，bool 走 else raise）；int 0 合法（slope=0）；leaky 缺省 0.01。
- **kaiming 的 a 缺省 0 ≠ calculate_gain 的 slope 缺省 0.01**：全默认场景 gain=√2（若错用 0.01 → √(2/1.0001)，bound 用例挂）。
- xavier：fi+fo 分母（[4,2,2,2] → fi=8/fo=16/和 24）；bound=√3·std。
- uniform_(a,b) 值映射：`a + (b-a)·(noise+1)/2`（noise∈[-1,1] → U(a,b)）；param 统一钉对称半宽（kaiming/xavier = bound，plain = (b-a)/2）。

## 变异记录（13 条全抓）

fan-receptive-drop、fan-swap、gain-tanh(5/3→1.5)、gain-slope-dflt(0.01→0)、gain-bool-allow（bool param 放行）、gain-unknown-ok（未知 nonlinearity 返回 1）、kaiming-a-default(a=0→0.01)、kaiming-bound-sqrt3、xavier-bound-sqrt3、xavier-fi-fo-max、mode-lower（大写 FAN_IN 场景挂）、uniform-map-formula、kaiming-normal-std-neg。

等价变异 2 条记 notes：xavier bound 官方路径 √3·std vs 直算 gain·√(6/(fi+fo))（ULP 容差吸收）；fan receptive field 乘法顺序反转（整数乘法交换律）。

## 红线

不调 torch.nn.init（solution 手写整条链）；trunc_normal_/orthogonal_/sparse_/dirac_ 不出；无全局 RNG 依赖（噪声显式入参）；与 `fw-distributed-sampler`（253）划界：本题是初始化公式链账，不是采样器装配。
