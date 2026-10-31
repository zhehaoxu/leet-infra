# Notes: UniPC 的 B(h) 预测-校正单步——两侧历史不同源

## 出处

- HuggingFace diffusers v0.42.1，`src/diffusers/schedulers/scheduling_unipc_multistep.py`：
  - 第 15 行的溯源注释写明实现来自论文与 `wl-zhao/UniPC`，并「modified based on `scheduling_dpmsolver_multistep.py`」。
  - `_sigma_to_alpha_sigma_t`：617-636（flow 分支返回 `(1-σ, σ)`，VP 分支返回 `1/√(σ²+1)` 与 `σ·α`）。
  - 预测侧 `multistep_uni_p_bh_update`：833-960。历史循环 897-904（`si = step_index - i` 在 898，`mi = model_output_list[-(i+1)]` 在 899，`rk` 在 902）；`D1s = torch.stack(D1s, dim=1)` 在 935；钉 0.5 在 937-938（`if order == 2`）；通式 `solve(R[:-1, :-1], b[:-1])` 在 940；两条更新式 945/952，收尾减项 950/957。
  - 校正侧 `multistep_uni_c_bh_update`：962-1097。历史循环 1032-1039，差别在 `si = step_index - (i+1)`（1033）而 `mi` 仍是 `[-(i+1)]`（1034）；钉 0.5 在 1075-1076（`if order == 1`）；通式 `solve(R, b)`（整阵，不划行列）在 1078；`einsum(rhos_c[:-1], D1s)` 在 1083/1091，`D1_t = model_t - m0` 与收尾 `x_t = x_t_ - α_t·B_h·(corr_res + rhos_c[-1]·D1_t)` 在 1086-1087 / 1094-1095。
  - 两侧的 `R`/`b` 递归相同：`h_phi_1 = expm1(hh)`、`h_phi_k = h_phi_1/hh - 1`，循环体里 **先** `factorial_i *= i + 1` **再** `h_phi_k = h_phi_k/hh - 1/factorial_i`（928-929 与 1063-1064）。
  - `step()`：1153-1233。`use_corrector` 1188-1190，校正调用 1194-1199（`order=self.this_order`），**缓冲左旋在 1201-1205**（`self.model_outputs[-1] = model_output_convert` 在 1205），`this_order` 的新值 1207-1213，预测调用 1217-1221。
  - `solver_type` 的别名映射在构造函数 272-276：`midpoint / heun / logrho` 被 `register_to_config(solver_type="bh2")` 永久改写，其余名字抛 `NotImplementedError`。
- 论文：Zhao, Bai, Rao, Zhou, Lu，"UniPC: A Unified Predictor-Corrector Framework for Fast Sampling of Diffusion Models"，arXiv:2302.04867（标题与作者经 arXiv API 核实）。

## 核心与易错点

- **两侧的「上一个输出」不是同一个张量。** 上游 1194 在校正时读的还是旋转前的缓冲，1201-1205 才把当前输出塞进 `model_outputs[-1]`。于是同一句 `mi = model_output_list[-(i+1)]`：校正侧的 `m0` 是 `step_index-1` 的输出、`i=1` 的差分项落在 `step_index-2`；预测侧的 `m0` 已经是当前输出、`i=1` 的差分项落在 `step_index-1`。同一条下标表达式在两侧指到不同的历史槽位，这就是「不同源」。题面把它摊平显式化：校正侧 `m0 = m_hist[-1]`、`m_i = m_hist[-(i+1)]`、`s_i = step-(i+1)`；预测侧 `m0 = m_t`、`m_i = m_hist[-i]`、`s_i = step-i`。写反配对在 `order>=3` 时必然移动结果；`order==2` 时两侧各只取一条差分、但槽位不同（校正侧 `m_hist[-2]`、预测侧 `m_hist[-1]`），写成同一个表达式照样会被测出来。另注意题面示例的 σ 是几何列、\(\lambda_i=(i-3)\ln 2\) 等距，于是 \(rk_i=-i\) 在两侧数值完全相同——能区分配对的只有历史槽位；`m_hist[-3]` 在 `(2,2)` 下根本不进公式，多塞或替换它必须逐位不变（`check_history_pairing_and_padding` 钉的就是这条）。
- **两处钉 0.5 的阶数不一样，而且不对称。** 预测侧省掉求解的是 `order_p == 2`（937-938），校正侧省掉求解的是 `order_c == 1`（1075-1076），常数都是 0.5。所以「阶数等于 solver_order」并不等于「走通式」。实测（`step_index=3`、题面示例输入）：`(2,2)` 下 `bh1`/`bh2` 的 `corrected` 逐位相同而 `prev` 差 0.095016；`(1,1)` 与 `(1,3)` 下 `corrected` 就差 0.075117、`prev` 差 0.047508。
- **`B_h` 在这套系数里几乎是个摆设。** `b_i = v_i·f_i/B_h`，通式解 `Rhos = b` 时 \(B_h\cdot\text{rhos}\) 把 \(B_h\) 约掉了，所以走通式的配置在 `bh1`/`bh2` 下逐位相同——实测 `(2,3) (3,3) (2,1) (3,1)` 四组两侧输出全部 `torch.equal`。只有钉住 0.5 的那两支让 \(B_h\cdot 0.5\) 真正落地。想验证自己没写错，最便宜的办法就是拿这两条不变式当金丝雀。
- **划行列只发生在预测侧。** 上游预测侧是 `solve(R[:-1, :-1], b[:-1])`（940）：矩阵少一行**也**少一列，因为最后一列对应 `rk_r = 1`，那一项的系数不进求解；校正侧是 `solve(R, b)`（1078），最后一条 `rhos_c[-1]` 单独乘在 `(model_t - m0)` 上（1086-1087）。少划一行一列会解出一个 3×3 的错系统，或者让 `rk=1` 那一列把矩阵搞成病态。
- **`factorial` 与 `h_phi_k` 的更新次序。** 上游 928-929 先 `factorial_i *= i+1` 再 `h_phi_k = h_phi_k/hh - 1/factorial_i`。在 `hh = -0.693147`（题面示例）下 \(v_2\) 的正确值是 −0.097989，写成 `1/f_i` 会得到 −0.597989，\(b_2\) 从 0.391958 变成 2.391958，\(B_h=-0.5\) 下的 `order_p=3` 解从 `[1.506568, -0.949263]` 变成 `[3.506568, -2.949263]`——差值恰好是 \(\pm2\)，因为 \(\Delta b=[0,2]\) 且 \(rk_1-rk_2=1\)。
- **终端 σ=0 不属于本题。** `final_sigmas_type="zero"` 时最后一步的 λ = +inf（α=1、σ̃=0），\(h=+\infty\)、\(hh=-\infty\)、\(\varphi_1=-1\)；`bh2` 有 \(B_h=-1\) 且一阶时差分项为空，于是直接落在 \(x_0\) 上；`bh1` 有 \(B_h=-\infty\)，一阶是 `0 * inf` 得 NaN，二阶以上是有限差分乘 ±inf 得 ±inf。题面因此要求 σ 严格为正并抛 `ValueError`，而不是让选手去凑这一格的退化分支。
- **`einsum("k,bkc...->bc...")` 需要至少三维的操作数。** 935/1070 的 `torch.stack(D1s, dim=1)` 对一维样本只给到二维，上游在这里直接 `RuntimeError`；本题把「样本至少二维、第 0 维是 batch」写进约束，省掉一个和算法无关的坑。
- **`this_order` 慢一拍。** 校正用的 `order` 是**上一步**算出来的 `self.this_order`（1194-1199 读，1207-1213 才写新值），`lower_order_final` 与 `lower_order_nums` 的暖机也在那几行。这套簿记已由 `diff-dpm-solver-order-schedule` 单独考过，本题把两个阶当显式入参给，避免同一道题里塞两件事。

## 与上游的差异

- 缓冲、`timestep_list`、`set_timesteps`、`step_index` 属性全部摊平：`m_hist / m_t / x_last / x_this / sigmas / step_index` 六个显式入参，历史配对由题面规定（见上），不再依赖「谁先跑谁后跑」。
- `convert_model_output`（ε/\(x_0\)/v 三套口径互转与 dynamic thresholding）和 `solver_p`（预测侧委托给另一个采样器的 UniP+UniC 变体，866-883 的提前返回）都不在本题内；入参已经是转换好的口径。
- `lower_order_final`、`disable_corrector`、`thresholding` 等构造期开关被删；`predict_x0` / `use_flow_sigmas` / `order_c` / `order_p` 成为每次调用的参数。
- 全程 float64 计算、float32 返回；上游按入参 dtype 走。别名 `midpoint/heun/logrho → bh2` 的映射保留（272-276），但映射发生在每次调用而不是构造期。
- 校正侧用 `x_last is None` 表示「本步不校正」（对应上游 `use_corrector=False`），此时 `corrected` 原样透传 `x_this`，而上游此时根本不产生这个中间量。

## 与其他题的分工

- `diff-dpm-solver-order-schedule`：`this_order`/`lower_order_final`/暖机的逐步簿记。本题两个阶都是入参。
- `diff-dpm-x0-conversion-threshold`：`convert_model_output` 的口径互转与动态阈值。本题接已转换的输出。
- `diff-dpmpp-2m`：DPM-Solver++(2M) 是**只有预测侧**的多步格式，没有校正，也没有「两侧历史不同源」这件事；本题的增量正是 P-C 这一对与它带来的下标错位。
- `diff-sigma-grid-construct` / `diff-noise-schedule`：σ 栅格与 ᾱ 表的构造。本题的 `sigmas` 是显式入参，只取 \(\lambda\) 一把。
- `diff-ddpm-step` / `diff-ddim-*`：一阶、单步。本题 `order_c=order_p=1` 时预测侧确实退化成 DDIM，这条闭式被 `check_first_order_closed_form` 用作交叉验证。
