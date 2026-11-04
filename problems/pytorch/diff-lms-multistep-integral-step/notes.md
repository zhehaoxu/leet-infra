## 回源

- `huggingface/diffusers` 的 `src/diffusers/schedulers/scheduling_lms_discrete.py`（675 行，Apache-2.0）。关键行：`step`（L545-623，convert 三分支 L587-597、导数与历史 pop L600-603、`order = min(self.step_index + 1, order)` L606、系数列表 L607、`zip(lms_coeffs, reversed(self.derivatives))` 落地 L610-612）；`get_lms_coefficient`（L267-294，`lms_derivative` 的 Lagrange 基连乘 + `scipy.integrate.quad(lms_derivative, sigmas[t], sigmas[t+1], epsrel=1e-4)`）；`scale_model_input`（L243-265，sample/√(σ²+1)，不入题）；`set_timesteps` 末位追加 0（L344）。
- 论文背景：线性多步法（Adams-Bashforth 族在 σ 空间的变体），Jolicoeur-Martineau et al. 2021（`@kohya` 实现的 k-diffusion 侧同名采样器 SpacedSampler 亦同构）。
- 与 `k-diffusion` 的 `sample_lms` 同源（Karras et al. 2022 附录），diffusers 版本把积分搬到了 VP σ 表上。

## 规范化（判题域 vs 官方）

- **scipy 依赖剔除**：官方 `integrate.quad(epsrel=1e-4)`；判题环境无 scipy，solution 用**幂系数展开解析积分**（ℓ_c 是 ≤3 次多项式，精确），oracle 用**16 节点 × 8 段复化 Gauss-Legendre 数值积分**——两条独立路径互验，quad 对光滑多项式实际精度 ~1e-12 远紧于其 epsrel 参数。
- **状态外置**：官方 `self.derivatives` 列表 + `self._step_index` 自增；本题 `state = {"derivatives": [...]}`、step_index 显式入参。历史长度契约显式化：调用前 `min(step_index, order)`、`step_index = 0` 必须传 None。
- **order 显式化**：官方 step 的 `order` 参数（默认 4）；本题入参，限 [1, 4]。
- 保留官方逐字公式：convert 三分支（**sample 分支 x0 = mo 也纳入**，比 754/755 多一支）、导数分母 σ、`zip` 的 reversed 配对、区间反向积分为负的 quad 语义。

## 题眼

1. **首步退化为欧拉**：order_used = min(step_index+1, order) 在 step 0 恒为 1，此时 ℓ_0 ≡ 1、coef[0] = ∫[σ0,σ1] 1 dτ = σ1 − σ0（**负值**），prev = sample − (σ0 − σ1)·derivative——与 522 `diff-euler-ancestral-step` 的确定性路径重合（但 LMS 无噪声）。order=1 的配置**全程**退化为欧拉（每步系数恒 σ_{t+1} − σ_t）。
2. **coeff[0] 配最新导数**：`zip(lms_coeffs, reversed(self.derivatives))`——历史追加在尾部、系数 c=0 是最新节点的 Lagrange 基。写成 `history[c]`（正序）只有 t < order 时恰好同序，t ≥ order 后立刻错位（变异 CAUGHT）。
3. **划分单位性**：Σ_c ℓ_c(τ) ≡ 1（Lagrange 基的基本性质）→ Σ_c coef[c] = ∫[σ_t, σ_{t+1}] 1 dτ = σ_{t+1} − σ_t（含末步积到 0 的 σ_{N−1} − 0）。这是与 oracle 独立的第二条校验线（用例直接断言）。
4. **反向区间语义**：σ 表递减 → 积分区间 [σ_t, σ_{t+1}] 反向 → 单个系数可正可负（order=2 时 c0 < 0、c1 > 0），但**总和恒为区间长度**（负值）。
5. **历史无 None 头**：与 756 SA-Solver 不同，LMS 历史从第一步起全是真实导数（首步就有一条），封顶淘汰用 pop 头部；契约长度 min(step_index, order) 是防历史错位的硬校验。

## 数值口径：quad 的替代

官方 `quad(epsrel=1e-4)` 的 epsrel 是**收敛判据**而非精度上界——对 ≤3 次多项式其实际误差 ~1e-12。solution 的幂系数展开解析积分是**精确的**（展开系数经 `denom` 累乘归一），两者差在 1e-12 量级，float32 输出容差 1e-6 下不可分辨。oracle 侧用复化 Gauss-Legendre（16 节点 × 8 段）独立数值积分，避免与 solution 共享「多项式展开」这条路径。

## 查重

- **522 `diff-euler-ancestral-step`**：LMS 首步/order=1 与 Euler 完全重合（本题用例显式钉死该退化），但 LMS 的本体是多步历史 + Lagrange 积分系数；Euler 的噪声拆分 LMS 不含。
- **754/755 KDPM2**：同样是「历史 + 系数」多步法，但 KDPM2 系数是解析 DPM 公式、交错表两调用状态机；LMS 是通用 Lagrange 积分、单调用。
- **`diff-deis-multistep-coef`**：DEIS 的系数也是积分（DEIS = Adam-Bashforth 的指数积分族），但 DEIS 在 λ 空间预计算系数表、无逐调用 σ 节点重算；LMS 每次调用用当前 σ 节点现算。
- **`diff-heun-predictor-corrector`**：单步导数平均，无历史。
- **756 SA-Solver**：同有 Lagrange 多项式，但 SA-Solver 是「Lagrange × 指数积分」的 sAB/sAM 带噪声状态机；LMS 是纯多项式积分的确定性多步法。

## 测试与变异

15 用例：oracle 全对拍（numpy 独立复刻 + 复化 Gauss-Legendre 数值积分，与 solution 的解析积分双路径互验）、首步欧拉退化（系数闭式 σ1 − σ0）、order=4 全循环（warmup 阶数爬升 [1,2,3,4,4] + 历史封顶）、order=2 封顶淘汰（首步导数被挤出）、state 契约（长度恰为 min(step_index, order)、篡改最早导数改变结果）、order-2 系数闭式与划分单位性（c0+c1 = σ2−σ1，含符号断言 c0<0<c1）、末步积到 0 收尾（系数和 = −σ_{N−1}）、v_prediction / sample 双分支全循环、六类 reject（prediction_type / sigmas 八种 / step_index 与 order / state 五种，全断言错误消息）、(1,2) 形状、两个 example（两步端到端 + order=1 全程欧拉）。

33 条变异 **33 CAUGHT + 0 等价**：sigmas 六查（长度/有限/非负/单调/末位 0/前 N 正）、step_index 双边与 bool、order 范围与 bool、prediction_type 白名单、state 长度契约/step0 拒绝/形状、convert 三分支（epsilon 符号、v_prediction 的 sample 项符号、sample 分支换 sample64）、导数公式与分母、pop 头部方向与封顶值、order_used 双公式（+2 与 cap −1）、σ 节点下标、积分上限、Lagrange skip 反转/分母符号/幂展开符号、积分幂次公式、落地配对序（正序 vs reversed）、落地出发点（sample vs pred_original）、落地符号。

## 踩坑

- **反向区间的符号直觉**：想当然认为「积分系数全负」——order=2 时 c0 = −36/7、c1 = +8/7（正负皆有），只有**和**恒为负的区间长度。断言先写错，用数值积分核对后改成符号断言 c0<0<c1 + 划分单位性。
- **`_close` 形状坑第三刷**：(1,1) numpy 数组 `float()` 炸、(1,2) 张量同炸——直接用 756 的 numpy reshape 版。
- **step 0 空 state 的契约决策**：`{"derivatives": []}` 在 t=0 数学上与 None 等价，但契约上拒绝（「step_index 为 0 时不能传 state」）——与 756 对齐，避免「合法输入两套写法」。
- 官方 `for k in range(order)` 里传入的 order 是 **min 后的 order_used**（L606 先改写局部变量再传 L607）——节点数随 warmup 递增，不是配置 order。
