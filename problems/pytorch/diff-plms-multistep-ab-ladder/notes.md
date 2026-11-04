## 回源

- `huggingface/diffusers` 的 `src/diffusers/schedulers/scheduling_pndm.py`（504 行，Apache-2.0）。关键行：`step_plms`（L324-392，ets 记账 L365-370、AB 阶梯 L372-384、`counter += 1` L387）；`_get_prev_sample`（L409-458，formula(9)：sample_coeff L446、denom L449-451、落地 L454-456，`prev_timestep >= 0 else final_alpha_cumprod` L431）；`set_timesteps` 的 skip_prk_steps 分支（L212-215，`concat([:-1], [-2:-1], [-1:])[::-1]`——**倒数第二格重复一次**）；`step` 的模式分派（L260-263，`counter < len(prk_timesteps)` 走 prk 否则 plms）。
- 论文：Pseudo Numerical Methods for Diffusion Models on Manifolds（Liu et al., arXiv 2202.09778）——plms 的 AB 阶梯 + formula(9)；prk 是其 4 子步 pseudo Runge-Kutta 热身。
- Stable Diffusion 的 `PNDMScheduler` 默认 `skip_prk_steps=True`（plms 全程），本题钉死该路径；prk 分支（12 次 counter 的 4 组子步 + `cur_model_output` 累加）不纳入判据。

## 规范化（判题域 vs 官方）

- **ᾱ 表显式化**：官方 `self.alphas_cumprod`（训练侧构造）+ `final_alpha_cumprod = alphas_cumprod[0]`；本题 `alphas_cumprod` 显式入参，表头即 final_alpha。校验 ∈ (0, 1]、单调不增——**ᾱ(0) ≈ 1−β₀ < 1**（真实 linear 表 0.99907），不是 1：ᾱ(0)=1 会使 formula(9) 在 t=prev 时分母为零。
- **counter / timestep / step_ratio 显式化**：官方 `self.counter` 自增、timestep 从 timesteps 表取、δ = num_train_timesteps // num_inference_steps；本题三个都入参。counter 决定分支（0/1/≥2）与 state 契约。
- **状态外置**：官方 `self.ets / self.cur_sample`；本题 `state = {"ets", "cur_sample"}`。调用前 ets 长度契约 [c1→1, c2→1, c3→2, c4→3, c≥5→4] 显式校验——这是官方「截断在调用内」记账的自然形状。
- **prk 剔除**：`step_prk`（L265-322）的 `cur_model_output` 累加、`prk_timesteps[counter//4*4]` 记账、12 次 counter 热身全部不纳入——判据与 SD 用法一致。
- 保留官方逐字公式：AB 四档系数（1/2、3/2、23/16/5、55/59/37/9）、counter 1 的 (t+δ, t) 重启对、formula(9) 的 denom 双 √ 结构。

## 题眼

1. **counter 1 的重启对**：counter 0 走半真步（t, t−δ）并暂存样本；counter 1 不 append，把两份输出梯形平均后**从暂存样本**在 (t+δ, t) 上重走整步——PNDM 论文「前两步 plms ≡ 一次梯形步」的记账核心。变异高发区（重启 / 不 append / 梯形公式各一条）。
2. **调用前 vs 调用后的 ets 长度**：截断 `[-3:]` 在调用内——调用前长度 [c1→1, c2→1, c3→2, c4→3, c≥5→4]，调用后 [1, 1, 2, 3, 4, 4, …]。首版把 c≥5 的调用前契约写成 3（截断后），被「长度必须恰为 3」的 reject 用例反噬——契约必须按「调用方持有的 state」记账。
3. **AB 阶梯的分支键是 append 后的长度**：len 1（counter 0/1）、2、3、≥4 四档；counter 2 的组合是 (3·m2 − m0)/2（ets = [m0, m2]，m1 不在历史里——counter 1 没 append！）。
4. **ᾱ(0) < 1 是 formula(9) 的生命线**：t == prev 时 α 差 = 0、sample_coeff = 1 → prev ≡ sample（恒等退化）；若 ᾱ(0) = 1 且 prev 落表头，denom = α_t·0 + √(α_t·(1−α_t)·1) 在 α_t = 1 时为 0——真实表 ᾱ(0) ≈ 0.999 避开。末步 prev_timestep = −δ < 0 → 表头。
5. **v_prediction 的转换时机**：ε = √α_t·mo + √(1−α_t)·sample 发生在 _get_prev_sample 内（记账之后）——counter 1 时 sample 已是 cur_sample，转换用的 sample 是重启后的样本，与 mo 的 timestep 不一致是官方原样行为。

## 查重

- **757 `diff-lms-multistep-integral-step`**：同为「历史 + 多步法系数」，但 LMS 的系数是 σ 空间 Lagrange 基**积分**（节点随 σ 表几何变化）、历史 pop 头部无特殊 counter 分支；PLMS 的系数是**固定有理数档位**（1/2、3/2、23/16/5、55/59/37/9）在 α 空间、带 counter 0/1 特例与 cur_sample 重启。
- **756 SA-Solver**：也有「上一步暂存样本/噪声」的三重时移，但 SA-Solver 是 sAB/sAM 双侧带噪声状态机、系数是 Lagrange × 指数积分；PLMS 单侧确定性、无噪声。
- **`diff-ddpm-step` / `diff-ddim`**：单步一族（无 ets 历史）；formula(9) 的落地公式与 DDPM ancestral 的后验均值同域但 PNDM 的 denom 结构不同。
- **`diff-deis-multistep-coef`**：DEIS 也是 AB 族的积分系数，但在 λ 空间且无 counter 0/1 特例。

## 测试与变异

13 用例：oracle 全对拍（numpy 独立复刻）、counter 0 恒等+暂存、counter 1 梯形+重启（prev_timestep = t 断言）、全循环 AB 阶梯爬升（调用后 ets 长度 [1,1,2,3,4]）、ets 封顶挤出（7 次调用后首条输出不在历史）、末步表头（t=0 → prev_timestep=−δ → α_prev=ᾱ(0)，且 t=prev 时 formula(9) 恒等退化 prev≡sample 手算钉死）、v_prediction 全循环、五类 reject（prediction_type / ᾱ 七种 / 标量十种 / state 十二种含 t+δ 越界，全断言错误消息）、(1,2) 形状、两个 example（两步梯形 + AB 系数线性组合闭式）。state 内容独立断言（ets 尾 = 本轮原始输出、counter 1 例外 = 上轮输出；cur_sample 只在 counter 0 非空）——防「oracle 吃 solve state」盲区（756 教训直接复用）。

31 条变异 **31 CAUGHT + 0 等价**：ᾱ 四查（长度/有限/值域/单调——单调 `>` 改 `>=` 由「相邻相等合法」用例反抓）、timestep 双边与 bool、step_ratio 下界、counter 非负与 bool、prediction_type 白名单、state 五查（c0 拒绝、长度契约、c1/c2 的 cur_sample、形状）、ets 记账三条（append 条件、截断宽度、append 对象）、AB 阶梯七条（梯形符号、重启丢失、2/3/4 阶系数各一、len2 分支键、c0 暂存）、timestep 记账两条（c1 的 prev、c0 的 prev）、表尾越界守卫、formula(9) 五条（表头、v_prediction 符号、sample_coeff 的 √、denom 删项、落地符号）。

## 踩坑

- **调用前 vs 调用后长度**：首版契约 c≥5→3（把截断记在调用前），被 reject 用例与全循环同时打脸——官方记账是「调用方持 4 条、调用内截断」，契约必须站在调用方视角。
- **ᾱ(0) 不能取 1**：测试首版用线性插值 ᾱ(t)=1−0.00098t 使 ᾱ(0)=1.0，末步手算时分母为零——换成真实 linear-β cumprod（np.cumprod(1−β)），ᾱ(0)=1−β₀≈0.9999。t=prev 的恒等退化反而成为干净的闭式断言（prev ≡ sample）。
- **counter 1 的 ets 尾**：不 append → ets[-1] 是**上一轮**输出——state 内容独立断言要分叉（counter 1 用 last_mo，其余用本轮 mo）。
- 基线红的假 CAUGHT：先加「相邻相等合法」用例时 timestep 越表尾导致基线 12/13，变异列表看起来全抓——修好基线后全部重跑（753 的教训原样复用）。
- skip 表的重复格：`[-2:-1]` 是**倒数第二个** timestep（N=5 时 500），不是 200——首版误读成 [-2] 重复了错误格子，表重算后 [999, 500, 500, 250, 0]。
