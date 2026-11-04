## 回源

- `huggingface/diffusers` 的 `src/diffusers/schedulers/scheduling_cosine_dpmsolver_multistep.py`（761 行，Apache-2.0；注意文件名 **dpmsolver 连写无下划线**）。关键行：`step` 的状态机（L604-686：`lower_order_final` 三条件 L641-646、`lower_order_second` L647-649、history 移位 L651-654、BrownianTree 噪声 L656-670、一阶/二阶分支 L672-675、`lower_order_nums` 封顶自增 L677-678）；`dpm_solver_first_order_update`（L448-487，SDE 噪声项 L484）；`multistep_dpm_solver_second_order_update`（L489-548，midpoint L533-539 / heun L540-546 两套 D1 系数）；`precondition_outputs` 的 EDM 预条件（L199-233，v_prediction 负 c_out L227）；`_sigma_to_alpha_sigma_t` 的 α≡1（L402-418）；`_compute_exponential_sigmas` / `_compute_karras_sigmas`（L306-363）；`precondition_noise` 的 `atan(σ)/π·2`（L182-197，timesteps 不进判题域）。
- 论文：Stable Audio Open（Evans et al., arXiv 2407.14358）与 EDM（Karras et al., arXiv 2206.00364）——EDM 预条件三元组 c_skip/c_out/c_in，σ 表是 karras/exponential 的**纯 σ 空间** schedule（无 ᾱ）。
- 结构承袭：两条 update 与 `scheduling_dpmsolver_multistep.py` 的 `dpm_solver_*_update` 同构，但本题是 **sde-dpmsolver++ 变体**（每步加噪声项）且 α_t ≡ 1（VE）。

## 规范化（判题域 vs 官方）

- **噪声显式化**：官方 `BrownianTreeNoiseSampler(model_output, σmin, σmax, seed)` 依赖 generator/seed 与张量形状，属管线侧；改为显式入参 `noise`，每次调用必需（官方两条 update 里 `assert noise is not None`）。
- **状态外置**：官方 `self.model_outputs / self.lower_order_nums`；装进 `state` dict 进出。`step_index` 官方内部自增（`_init_step_index`/`index_for_timestep`），本题显式入参——`timesteps`（atan 变换）与 `index_for_timestep` 的查表逻辑不进判题域。
- **配置显式化**：`solver_order / solver_type / lower_order_final / euler_at_final / final_sigmas_type / sigma_data / prediction_type` 全部入参，默认值 = 官方 config 默认。
- **表构造不内联**：`sigmas` 原样入参（exponential/karras 生成 + `sigma_last` 追加留给调用方），λ 逐点现算。
- **solver_order 白名单**：官方无校验（=3 时 `prev_sample` 未绑定直接 UnboundLocalError）；本题限定 {1, 2}。
- **官方 elif 恒真的合并**：`elif solver_order == 2 or lower_order_nums < 2 or lower_order_second` 在 solver_order ∈ {1, 2} 校验下恒真（`lower_order_second` 是死代码），实现合并为 else，docstring 记录。
- **h=0 / h_0=0 guard**：官方在相邻 σ 相等时产出 0/0（torch 给 nan/inf 垃圾）；本题抛 `ValueError`——判据边界，官方各 schedule 相邻相等只出现在 sigma_min 收尾的末位。
- **float64 语义保留**：σ = 0 时 λ = +∞（`math.exp(−inf) = 0`，与 torch 一致），一阶 update 三系数 {0, 1, 0} 逐字。

## 题眼

1. **λ 只以差值 h 出现** → α 的任何固定缩放平移相消（`_sigma_to_alpha_sigma_t` 改 `σ·0.5` 后 h 逐位不变，见等价变异表）。本题的恒等式红利：**e^{−h} = σ_t/σ_s0**，1 阶 sample 系数 = (σ_t/σ_s0)²、decay = 1 − (σ_t/σ_s0)²，示例可全有理化（σ 取 3 与 1）。
2. **EDM 预条件在 convert**：c_skip/c_out 都依赖 `sigma_data`，v_prediction 的 c_out 是**负**的（v = x0 − σ·ε 的符号约定）——只测 σd=1 会漏掉 epsilon 分支的 d 依赖（变异 `m-cout-eps` 首轮 SURVIVED 的原因）。
3. **降阶三条件是「或」**：`euler_at_final` / (`lower_order_final` and N<15) / `final_sigmas_type == "zero"` 任一为真，末步（且仅末步）降 1 阶。N ≥ 15 时只有 euler 或 zero 能降——三个分支各出一条变异，各自需要不同的配置才能抓到（heun + sigma_min 收尾使错误二阶在 h=0 处炸出可观测差异）。
4. **官方永远不让二阶碰 σ_t = 0**：zero 收尾使末步 flag 无条件为真；若变异破坏这一点，二阶路径 h = +∞ → r0 = 0 → D1 除零炸——`m-lof-zero` 的抓法。
5. **末步 zero 收尾的恒等步**：prev ≡ x0（系数 {0,1,0}）；sigma_min 收尾且降阶时 prev ≡ sample（h=0，系数 {1,0,0}）——两个恒等步都用独立断言钉死。
6. **中段冷启动合法**：`state=None` + 任意 `step_index`（官方 `begin_index` 场景）：首调 1 阶，history 的 None 头随移位被推出，第二次调用 m1 正是首调的 x0。
7. **state 不变量**：前 solver_order−lo 条 None、末 lo 条真实张量；`1 ≤ lo ≤ min(solver_order, step_index)`——lo ≥ 1 隐含「给 state 必须至少调用过一次」，因此 step 0 传 state 无需专门检查即被 lo 范围拒绝（不要写冗余的显式检查，否则变异等价）。

## 查重

- **`diff-dpm-solver-order-schedule`**：同为 DPMSolverMultistep 族的阶数调度。但那边是 αβ 离散约定（ᾱ 表 + eps/x0 转换公式）、无 SDE 噪声项、无 EDM 预条件、σ 表由 ᾱ 推出；本题是 EDM/VE 约定（α≡1、纯 σ 空间 exponential/karras 表 + 0/sigma_min 收尾）+ BrownianTree SDE 噪声 + c_skip/c_out 预条件 + 降阶三条件（多出 `euler_at_final` 与 `final_sigmas_type` 两分支）。
- **`diff-sasolver-adams-pc-step`（756）**：同为随机 multistep。但 SA-Solver 是 sAB/sAM 双侧结构（corrector 回头修正 + 三重时移状态）、系数来自 Lagrange × 指数积分、τ 是连续旋钮；本题固定 1/2 阶单侧 predictor、系数是有限差分外推（r0 一项）、噪声强度不可调。
- **`diff-kdpm2-ancestral-noise`（755）/ `diff-euler-ancestral-step`（522）**：单步 ancestral 噪声拆分，无多步 history 与 r0 外推。
- **`diff-dpmpp-2m` / `diff-deis-multistep-coef` / `diff-lms-multistep-integral-step`（757）**：确定性 multistep（无噪声项），系数构造完全不同（2M 的 Cholmod 式闭式 / DEIS 与 LMS 的积分系数）。
- **`diff-plms-multistep-ab-ladder`（758）**：AB 阶梯 + ets 记账，无 SDE 噪声、无 EDM 预条件。

## 测试与变异

15 用例：oracle 全对拍（numpy float64 独立复刻状态机，逐步比较 prev/x0/r0/d1c/this_order/state 内容与 None 模式）、首调默认路径（1 阶 + 系数闭式独立验证 + state 头 None）、epsilon vs v_prediction（含 σd=2 的 epsilon——堵 `m-cout-eps` 盲区）、step 1 二阶 midpoint（r0>0 + d1c 闭式 + state 双真实历史）、step 1 heun（d1c ∈ (0,1) 闭式）、默认 N=6 全循环（末步 prev ≡ x0 独立断言）、heun + sigma_min 收尾 N=10 全循环（末步 h=0 恒等 prev ≡ sample）、euler_at_final N=15、zero 收尾 N=15 + 双开关关（`"zero"` 分支独立生效）、solver_order=1 全程 1 阶、karras 表全循环、sigma_data=2 闭式、中段冷启动（begin_index 场景）、h=0 二阶拒绝、state 契约 10 种（全部断言错误消息防「空抓」）、入参校验 21 种。每个全循环都带**不依赖 oracle 的独立断言**（lo 计数 = min(i+1, solver_order)、一阶条件独立重推、末位 history = 本步 x0——756 的对拍盲区教训直接复用）。

48 条变异 **47 CAUGHT + 1 等价**：

| 等价变异 | 实证 |
|---|---|
| `m-alpha`（`_sigma_to_alpha_sigma_t` 返回 `(1, σ)` → `(1, σ·0.5)`） | 300 组随机配置（N 2-15、双 solver_order/solver_type/prediction_type/final_sigmas_type、随机 lof/euler/σd）全循环逐位 0 差异（含两侧同抛 ValueError 的异常路径对齐）。结构恒等式：λ 只以差值 h 进入公式，α 缩放的 log 2 平移相消 |

其余 47 条覆盖：sigmas 六查（长度/有限/负值/末位外全正/单调/张量性）、step_index 双边与 bool、solver_order 范围与 bool、solver_type / final_sigmas_type 白名单、lof/euler 布尔性、sigma_data 三查、noise 必需与形状、state 七查（键集/长度双向/lo 下界与 step 上界/None 头/真实尾/形状）、c_skip 公式、epsilon 与 v_prediction 的 c_out（后者含 σd 因子）、convert 的 σ 下标、prediction_type 报错文案、降阶四变异（step 条件位移 / euler 分支删除 / N<15 松紧 / zero 分支改写）、warmup cap（`lo < 1` → `< 2`）、solver_order 判定、1 阶四变异（(σt/σs0) 因子 / exp 符号 / decay 指数 / 噪声 σ 下标）、2 阶七变异（r0 交换 / D1 加法 / m1 下标 / midpoint 0.5 / heun 分母符号 / h guard / σ_s1 下标）、lo 封顶条件、history 移位方向、λ 三变异（inf 分支 / 符号 / α 缩放）。

## 踩坑

- **变异前基线确认再验证一次**（758 教训复用）：`check_full_loop_default` 首版自断言把 step 0 的 1 阶 warmup 当错误（`outs[:-1]` 全断言 2 阶）→ 基线 14/15 红。若当时直接跑变异会得到整片假 CAUGHT——先修断言（`outs[0]` 是 1 阶）再重跑基线，15/15 绿后才开变异。
- **`exp(log(0.3))` 不保证逐位等于 0.3**：sigma_min 收尾用例的「末位」必须**精确复制** `core[-1]` 的浮点值再拼接，否则 h ≈ 1e-16 量级的抖动会让「恒等步 prev ≡ sample」断言脆弱（数值上仍过 1e-6 容差，但 h=0 guard 用例会翻车——h 万一为正微小值，二阶不再拒绝）。
- **随机实证的组合必须自洽**：首版生成器随机选 `fst` 与表尾，造出「fst=sigma_min 但表尾 0.0」的官方不可能组合 → 末步二阶遇 σ_t = 0 → h = +∞ → r0 = 0 → `1/0` ZeroDivisionError（不是 ValueError，逃出 try）。官方永远通过 flag 把零收尾末步压到 1 阶——生成器改为 fst 与表尾联动 + 异常路径对齐后实证通过。
- **判题域张量全 float64**：σ 跨 3 个数量级（0.3 → 500），float32 往返的 λ 相对误差会逼近 1e-6 对拍容差——本题判题入参一律 float64（判题器原样透传张量，已核实 judge 无 dtype 强转）。
- **λ 的 σ=0 分支**：`math.log(0)` 抛 ValueError 而 `torch.log(0) = −inf`——λ helper 必须 `σ == 0 → math.inf`，`math.exp(−inf) = 0.0` 与 torch 语义对齐；否则 zero 收尾末步直接崩。
