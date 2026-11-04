## 回源

- `huggingface/diffusers` 的 `src/diffusers/schedulers/scheduling_sasolver.py`（1423 行，Apache-2.0）。关键行：`step` 的五拍状态机（L1262-1362，corrector 条件 L1300、history 移位 L1315-1320、warmup L1329-1339、`last_sample/last_noise` 暂存 L1341-1342）；`stochastic_adams_bashforth_update`（L999-1070，含 order-2 修正 L1024-1045）；`stochastic_adams_moulton_update`（L1139-1206，含 order-2 修正 L1164-1181）；`get_coefficients_exponential_positive`（L709-769）与 `lagrange_polynomial_coefficient`（L771-892）；`_sigma_to_alpha_sigma_t`（L436-455）；`set_timesteps` 的 sigmas 末位（L336-337，非 karras 路径追加 `sigma_last = ((1-ᾱ_0)/ᾱ_0)^0.5`，**微小正值而非 0**）。
- 论文：SA-Solver: Simplifying and Stabilizing Diffusion Sampling（Xue et al., arXiv 2309.05019）——SA-Predictor = 随机 Adams-Bashforth（sAB）、SA-Corrector = 随机 Adams-Moulton（sAM），τ 是 SDE 噪声强度旋钮（τ=0 退化为 ODE 求解器）。
- 与 UniPC 的关系：SA-Solver 代码基于 DPM-Solver-multistep 改写，order-2 修正是 UniPC 同款的 O(h³) 提质项——但两侧系数生成方式完全不同（见查重）。

## 规范化（判题域 vs 官方）

- **噪声显式化**：官方 `randn_tensor(generator)` 每步抽取；改为显式入参 `noise`，每次调用必需（τ=0 时噪声项为 0 但仍暂存 `last_noise`——与官方状态机一致，下一步 corrector 消费它）。
- **τ 拆双参**：官方 `tau_func(timestep)` 在 corrector 处用**移位前**的 `timestep_list[-1]`（= 上一步 timestep）、在 predictor 处用**移位后**的（= 当前 timestep）。判题域无内嵌时钟，拆成 `tau_corrector`（上一步取值）与 `tau`（当前步取值）两个显式入参，校验非负且有限。
- **状态外置**：官方 `self.model_outputs / self.last_sample / self.last_noise / self.lower_order_nums / self.this_corrector_order`；全部装进 `state` dict 进出。`step_index` 官方内部自增，本题显式入参。
- **orders / lower_order_final 显式化**：官方 config 常量；本题入参，warmup 公式逐字保留。
- **表构造不内联**：与 754/755 不同，本题没有辅助 σ 表——`sigmas` 原样入参（官方 set_timesteps 产出全为正的表），λ 逐点现算。
- 保留官方逐字公式：双 log 的 λ 路径（不换成 −log σ）、`exp`/`sqrt` 的非合并形式、predictor/corrector 两套**不同**的 order-2 修正。

## 题眼

1. **三重时移**：corrector 消费的三样东西全部来自**上一步**——`this_corrector_order`（上一步 warmup 算出存下的）、`last_sample`（上一步 corrector 后、predictor 前的样本）、`last_noise`（上一步 predictor 注入的噪声）。predictor 用的全是**本轮**的（阶数 tp、样本 corrected、噪声 noise）。变异高发区。
2. **两侧历史差一格**：corrector 的模型输出表 = 移位前 history + [x0]（x0 是「刚拿到、还没入库」的），predictor 的 = 移位后 history（x0 已在末位）。M = max(predictor_order, corrector_order − 1) 的 **−1 正是给 corrector 的 x0 腾位**。
3. **λ 恒等式**：VE 约定下 λ = log α − log(σα) ≡ −log σ、e^λ ≡ 1/σ。两个直接红利：① 1 阶系数（τ=0）有闭式 coefs[0] = e^{λ_t} − e^{λ_s0} = 1/σ_target − 1/σ_source；② 手算示例可以全有理化（σ 取勾股数 4/3、3/4 使 √(σ²+1) 为有理数）。官方双 log 路径数学冗余但逐字保留作判据。
4. **两套 order-2 修正不同源**：predictor 版（L1024-1045）的 Δ 分母是 `(λ_s0 − λ(sigmas[step_index−1]))`（相邻 λ 差），且分子是 `h²/2 − …/(1+τ²)²`（无 h）；corrector 版（L1164-1181）分子是 `h/2 − …/((1+τ²)²·h)`（有 h）。照抄时极易互串——各出一条变异。
5. **warmup 的两个 cap**：`tp ≤ lower_order_nums + 1`、`tc ≤ lower_order_nums + 2`（corrector 多 1 是因为它消费的是「上一步算好的阶数」，天然晚一步生效）。与 M 的配伍保证 None 头永不被读：tp−1 ≤ min(i, M)、tc−1 ≤ min(i, M)（tc ≤ corr ≤ M+1 由 M 的定义保证）。`lower_order_final` 的 `N − step_index` 使**末步 predictor 必降 1 阶**。
6. **中段冷启动合法**：`state=None` + `step_index>0` 对应官方 image-to-image（`last_sample is None` → `use_corrector=False`），此时无 corrector、warmup 从 1 阶重新爬。
7. **官方表全正**：SASolver 各 schedule 的 sigmas 无 0 收尾（非 karras 末位追加的是 σ(t=0) 微小正值；karras/exponential/flow 末位重复最后一个正值）——校验「全为正」有据，λ 永不 −inf。

## 对拍盲区：oracle 吃 solve 的 state

多步循环对拍的写法是「每步把 solve 返回的 `state` 原样透传给 solve 与 oracle」——于是 **state 内容错了对拍照样绿**：`m-state-last-sample`（把 `last_sample` 从 corrected 换成本步输入 sample）首轮 SURVIVED，19/19 全过。corrector 拿着错的 `last_sample` 算出错的 corrected，但 oracle 也拿同一个错 state 算出同样的错值。修复：循环里加**独立 state 内容断言**（不经过 oracle）——`state.last_sample == 本步 corrected_sample`（无 corrector 时 == 本步输入 sample）、`state.last_noise == 本步入参 noise`、`state.model_outputs[-1] == 本步 pred_original_sample`——补上后 4 个循环用例齐抓。教训与 754 的「变异前先确认 selftest 绿」同级：**对拍类测试要区分「被测值」与「透传值」，透传值必须另设不依赖 oracle 的断言**。

## 查重

- **`diff-unipc-pc-step`**：同为 predictor-corrector 双侧结构。但 UniPC 的系数来自**线性方程组求解**（rhos = solve(R, b)，Vandermonde 型），本题系数是 **Lagrange 多项式 × 闭式指数积分**（手写 0-3 次公式）；UniPC 无 warmup 状态机（阶数是调用期常量）、无噪声注入/τ、无「上一步阶数」三重时移。两侧历史不对称的机制也不同（UniPC 是 λ 下标差一格，本题是移位时机差一拍）。
- **`diff-deis-multistep-coef` / `diff-dpm-solver-order-schedule`**：单侧 predictor 族，无 corrector、无噪声。
- **`diff-heun-predictor-corrector`**：单步导数平均式 PC，无多阶 history、无 λ 节点插值。
- **754/755 KDPM2**：两调用交错 σ 表状态机，单阶导数，无多步历史。
- **`diff-euler-ancestral-step`（522）**：单调用 Euler ancestral，噪声拆分同域但无 corrector/多阶。

## 测试与变异

19 用例：oracle 全对拍（numpy float64 独立复刻五拍状态机，逐步比较 prev/corrected/x0/双侧系数/双侧阶数）、首调 warmup（tp=1、无 corrector、state 三件套内容）、step 1 corrector 首次生效、默认 2/2 阶全循环（含末步降阶断言）、τ=0 ODE 路径（两套噪声逐位一致）、3/3 与 4/4 阶全循环（覆盖 Lagrange 2/3 次与 3 阶指数积分）、`lower_order_final=False`（末步不降阶）、v_prediction 全循环、中段冷启动、karras 式末位重复（h=0 → 系数/噪声项全 0，prev ≡ corrected）、(1,2) 形状端到端、state 契约（篡改 `last_noise`/`this_corrector_order` 改变结果 + state 内容独立断言）、六类 reject（prediction_type / sigmas 七种 / step_index 四种 / orders 与 flags / noise 与 tau / state 十二种，全部断言错误消息防「空抓」）。

44 条变异 **43 CAUGHT + 1 等价**：

| 等价变异 | 实证 |
|---|---|
| `m-corr-cond`（corrector 条件 `True` → `step_index > 0`） | 300 组随机配置（步数 2-6、阶数 1-4、lof、双 prediction_type、τ 序列）全循环逐位 0 差异。结构恒等式：state 分支内 `step_index == 0` 已被前置校验拒绝，条件恒真 |

其余 43 条覆盖：sigmas 五查（长度/有限/全正/单调/张量性，含全正 vs 非负的边界——官方表无 0 使「允许 0」的松动会在 λ=−inf 处炸出 inf/nan）、step_index 双边、orders 范围与 bool、`lower_order_final` 布尔性、prediction_type 白名单、τ 三查（非负含等号的边界——`< 0` 改 `<= 0` 会把 τ=0 的 ODE 路径拒掉）、noise 必需与形状、corrector 条件、corrector σ/节点下标、两套 order-2 修正（跳过/公式/分母各一条）、corrector 噪声源与出发点、history 移位方向与追加值、`lower_order_nums` 递增条件、两个 warmup cap、`lower_order_final` 双公式、state 的 last_sample/last_noise 暂存值、α 公式、λ 符号、convert 双分支、predictor σ 下标、历史取序（`-(k+1)` vs `k`）、gradient 因子、noise_part、x_t 系数、指数积分 1/2 阶常数、Lagrange 符号。

## 踩坑

- **`this_corrector_order` 的两个身份**：返回 dict 里的 `this_corrector_order` 是**本轮 warmup 算出的新值**（存给下一步用），而 corrector 系数列表长度是**上一步存下来的旧值**——对拍断言按「当次实际用的阶数」数长度，首轮写成 `range(this_corrector_order)` 在 3 阶配置下 IndexError。
- **末步降阶 vs 手算直觉**：N=2 的表在 step 1 手算 tp=2，实际 `lower_order_final` 使 tp = min(2, N−1) = 1——示例数值先算错一版，回看官方 L1329-1330 才对上。
- **oracle 必须传 solve 的 state**（torch 张量），不能传 oracle 自己的 numpy state（与 755 同款纪律）；`_close` 需兼容 (1,2) 形状（早期 `float()` 写法在多元素张量上炸）。
- **`steps=1` 不是合法判题域**：sigmas 长度 2 被「N ≥ 2」校验拒绝——随机实证脚本的步数下限要从 2 起。
- **h=0 边界**：karras 末位相邻相等时，末步 predictor h=0 → 系数与噪声项全 0、prev ≡ corrected（数学恒等，用例钉死）；但 `lower_order_final=False` + 中段相邻相等会让 order-2 修正除以 h=0（Python float 除零抛 ZeroDivisionError，官方 torch 给 nan）——判据边界，判题域不含此组合（官方各 schedule 相邻相等只出现在末位）。
