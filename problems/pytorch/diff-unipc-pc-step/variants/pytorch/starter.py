def solve(m_hist, m_t, x_last, x_this, sigmas, step_index, order_c=1, order_p=1,
          predict_x0=True, solver_type="bh2", use_flow_sigmas=False):
    """UniPC 的 B(h) 预测-校正单步（diffusers `UniPCMultistepScheduler` 的两支更新装配）。

    语义约定
    --------
    一次调用复现上游 `step()` 里「先校正、再预测」的那一段，输入全部显式给出：

    - `m_hist`：**已转换**成求解器变量的历史模型输出，从旧到新排布。
      `m_hist[-1]` 属于 `step_index - 1`，`m_hist[-k]` 属于 `sigmas[step_index - k]`。
      校正侧把它当作旋转前的缓冲（`m0 = m_hist[-1]`，第 i 项取 `m_hist[-(i+1)]`），
      预测侧则等价于「先压入当前输出再旋转」，于是 `m0 = m_t`、第 i 项取 `m_hist[-i]`。
      两侧的下标配对不同，是最容易写错的地方。
    - `m_t`：当前 `step_index` 处的已转换输出。上游的 ε/x₀/v 口径转换不在本题范围。
    - `x_last`：上一条预测**之前**的样本（对应 `sigmas[step_index - 1]`）。传 `None`
      表示关闭校正（上游的 `step_index == 0`、`disable_corrector` 或 `last_sample is None`
      三种情形合并成这一个开关），此时输出第一项原样返回 `x_this`。
    - `x_this`：上一条预测**之后**的样本，即当前样本（对应 `sigmas[step_index]`）。
    - `sigmas`：一维正数张量，长度至少 `step_index + 2`；终端 `σ = 0` 的那一步不在范围内。
    - `step_index`：当前的步序号（int，从 0 起）。
    - `order_c` / `order_p`：校正阶与预测阶。上游的 corrector 用的是**上一步**算好的
      `this_order`，predictor 用的是本步新算的，二者在同一步里本就可能不同；
      阶数调度账已由合集内另一题负责，这里一律显式给出。

    公式（中间量在 float64 下算）
    ----------------------------
    `alpha(σ), sigma_tilde(σ)`：`use_flow_sigmas` 为真时 `(1 - σ, σ)`，
    否则 `alpha = 1/sqrt(σ² + 1)`、`sigma_tilde = σ * alpha`。记 `λ = log(alpha) - log(sigma_tilde)`。

    两侧各自取 `(s0, t)`：校正侧 `s0 = step_index - 1, t = step_index`；
    预测侧 `s0 = step_index, t = step_index + 1`。`h = λ_t - λ_s0`，
    `hh = -h`（`predict_x0=True`）或 `h`（False），`phi_1 = expm1(hh)`，
    `B_h = hh`（`bh1`）或 `expm1(hh)`（`bh2`）。

    对 `i = 1 .. order - 1`：`rk_i = (λ_{s_i} - λ_{s0}) / h`，`D1_i = (m_i - m0) / rk_i`；
    `rks = [rk_1, ..., rk_{order-1}, 1]`。
    递推：`h_phi_k = phi_1 / hh - 1`，`factorial_i = 1`；对 `i = 1 .. order` 依次
    `R_i = rks ** (i - 1)`、`b_i = h_phi_k * factorial_i / B_h`，
    然后 **先** `factorial_i *= i + 1`，**再** `h_phi_k = h_phi_k / hh - 1 / factorial_i`。

    `rhos` 的取值有两处不对称的特例（都是常数 0.5）：
    预测侧 `order_p == 2` 时用 `[0.5]`，`order_p >= 3` 时解 `R[:-1, :-1] x = b[:-1]`，
    `order_p == 1` 时整条差分项不存在；
    校正侧 `order_c == 1` 时用 `[0.5]`，`order_c >= 2` 时解 `R x = b`（**不**去掉末行末列）。

    合成：校正侧 `acc = Σ rhos_c[:-1]·D1_i + rhos_c[-1]·(m_t - m0)`；预测侧
    `acc = Σ rhos_p·D1_i`。最后
    `x_t = (σ_t/σ_s0)·x - α_t·phi_1·m0 - α_t·B_h·acc`（`predict_x0=True`），
    或 `x_t = (α_t/α_s0)·x - σ_t·phi_1·m0 - σ_t·B_h·acc`（False）。
    校正侧的 `x` 是 `x_last`，预测侧的 `x` 是校正后的样本。

    返回
    ----
    `(corrected, prev)`：两个与 `m_t` 同形状的 float32 张量；`x_last` 为 `None`
    时第一项就是 `x_this`。

    非法输入一律抛 `ValueError`
    --------------------------
    非张量、张量少于二维、形状不一致、含 NaN/inf；`sigmas` 不是一维、含非正数、长度不足；
    `use_flow_sigmas` 下 σ 不在 (0, 1) 导致 λ 不有限；`step_index` / `order_c` / `order_p`
    不是 Python int（`bool` 非法）或小于下界（order ≥ 1）；启用校正但 `step_index < order_c`
    或历史不够长；预测侧 `step_index < order_p - 1` 或历史不够长；`predict_x0` /
    `use_flow_sigmas` 不是 bool；`solver_type` 不在 `bh1`/`bh2`（`midpoint`/`heun`/`logrho`
    按上游规则静默映射成 `bh2`，其余名字报错）。

    禁止 `import diffusers`。
    """
    raise NotImplementedError("请实现 solve()")
