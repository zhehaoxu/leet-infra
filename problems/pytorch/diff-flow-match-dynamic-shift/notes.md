# 详解：Flow Matching 的动态时间平移与 terminal 拉伸

## 出处

HuggingFace diffusers 的 `src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py`：`time_shift`（分派到 `_time_shift_exponential` / `_time_shift_linear`）、`stretch_shift_to_terminal`、以及 `set_timesteps` 里第 2–6 步的次序（平移 → 拉伸 → 转换器 → 乘 `num_train_timesteps` → 追加终端）。`shift_terminal` 的出处是 LTX-Video 的 `ltx_video/schedulers/rf.py`，代码注释里给了链接。动态平移的 \(\mu\) 在 Flux 一类 pipeline 里由分辨率算出（`base_shift` / `max_shift` / `base_image_seq_len` / `max_image_seq_len`），那部分已由 `diff-rf-timestep-shift` 覆盖；本题只吃「已知 \(\mu\)」之后的步骤。

## 核心与易错点

1. **两条公式其实是同一族**：静态平移 \(s\cdot x/(1+(s-1)x)\)、线性平移 \(\mu x/(1+(\mu-1)x)\) 在 \(\mu=s\) 时完全相等；指数平移在 \(\mu=\ln s\) 时与它们相等。上游之所以写三条，是因为 Flux 的 \(\mu\) 由分辨率插值得到、Dev 用静态 shift、SD3.5 用指数型。实现里可以只写 `e^\mu/(e^\mu+x)` 与 `\mu/(\mu+x)` 两支，但**指数支的尺度是 \(e^\mu\) 而不是 \(\mu\)**，把两者写成同一个 `mu` 是本题最常见的公式错。
2. **端点被钉住**：\(x=1\) 时 \((1/x-1)=0\)，任何平移都映到 1；\(x=0\) 时 \(1/x=\infty\)，映到 0。所以 `sigma_max=1, sigma_min=0` 时首末 σ 不变，只有中间被推动。用 \(1/x\) 实现时 \(x=0\) 会触发除零：torch 里得到 `inf` 并自然算出 0，纯 Python oracle 必须显式短路。
3. **`stretch_shift_to_terminal` 的次序**：`scale = (1 - z[-1]) / (1 - shift_terminal)` 里的 `z[-1]` 必须是**平移之后**的末项。若先拉伸再平移，平移会再次改变末项，`shift_terminal` 就落空。测试里专门构造了「先拉伸再平移」的错序结果，要求它与正确结果的末项相差 >1e-3。
4. **timesteps 用拉伸后的 σ**：`timesteps = sigmas * T` 在拉伸之后计算，所以 `timesteps[-1] == shift_terminal * T`，而不是拉伸前的 `0`。
5. **`invert_sigmas` 会重算两次**：先 `sigmas = 1 - sigmas`，再 `timesteps = sigmas * T`（不是取负或复用旧 timesteps），并且终端追加 **1** 而不是 0。反相后 σ 单调递增，第一个元素是 0、最后一个真实元素是 `1 - shift_terminal`（若设了 terminal）。
6. **\(N=1\) 与 terminal 不能共存**：只有一个 σ 时 `1 - z[-1]` 为 0，`scale = 0/(1-terminal)`，`(1-z)/scale` 是 `0/0` → NaN。上游不拦，本题在参数校验处直接抛 `ValueError`，并在题面写明。
7. **`linear` 平移要求 \(\mu>0\)**：\(\mu=0\) 时分子恒为 0，整串 σ 归零；指数支 \(\mu=0\) 则是恒等变换（\(1/(1+x)\) 乘 \(1\) 后仍是 \(e^0/(e^0+x)\)）。两者对 \(\mu=0\) 的行为不对称，所以只对线性支设下界。
8. **终端值可能和末项重复**：`sigma_min=0` 时末项本就是 0，追加的终端也是 0，`sigmas` 末尾出现两个 0。这是上游的正常形态，不是漏写。
9. **σ 反相不是对合**：`invert_sigmas` 在追加终端之前生效，所以「反相再反相」并不回到原序列（终端值会从 0 变成 1 再变回来，但中间项与 timesteps 的推导路径已经不同）。测试不对合验证，只逐项对拍。

## 对拍口径

判题用纯 Python 标量 oracle（`_oracle`）逐项重做 linspace / 平移 / 拉伸 / 反相 / 终端，再与参考解比对：

- **示例值**：\(N=4\)、六种配置（静态、指数、线性、terminal、invert、terminal+invert）的 timesteps 与 sigmas 全部六位小数硬编码；
- **等价性**：静态 shift 与线性 \(\mu=\)shift、指数 \(\mu=\ln\)shift 在 \(N\in\{2,4,7\}\)、shift\(\in\{1,1.5,2,5\}\) 上逐元素一致；shift=1 与指数 \(\mu=0\) 都是恒等；
- **单调与端点**：三种平移 + terminal 组合下 σ 单调不增（无 terminal 时首末固定为 1 和 0），timesteps 恒等于 `sigmas[:-1]*T`；
- **terminal 次序**：末项等于 `shift_terminal`、整体等于 \(1-(1-z)/\text{scale}\)、timesteps 末项为 `terminal*T`，并断言错序写法末项偏差 >1e-3、首项被推到更靠噪声侧；
- **反相**：`sigmas[:-1] == 1 - 原 sigmas[:-1]`、终端为 1、timesteps 按反相后重算、单调递增；
- **σ 区间与 \(N=1\)**：`sigma_min=0.1/sigma_max=0.9` 的六位小数结果、\(N=1\) 仅一个 timestep；
- **标尺**：`num_train_timesteps ∈ {1,10,1000,4096}` 时 timesteps 严格按比例缩放；
- **随机/组合对拍**：\(N\in\{1,2,3,5,8\}\) 下 8 组配置（含负 \(\mu\)、窄 σ 区间 + terminal + invert）与 oracle 逐项一致；
- **非法输入**：18 组（\(N\) 为 0/浮点/bool、shift 非正或 NaN、未知 shift 类型、动态平移缺 \(\mu\)、\(\mu=\infty\)、线性 \(\mu=0\)、terminal 为 0/1/负数、\(N=1\) + terminal、σ 上下界非法、\(T=0\)）都必须抛 `ValueError`。

共九组断言。

## 复杂度

\(O(N)\) 时间与空间，只做两次逐元素代数变换与一次累加。
