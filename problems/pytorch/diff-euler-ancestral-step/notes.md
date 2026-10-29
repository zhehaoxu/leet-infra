# 详解：EulerAncestral 的 σ_up/σ_down 拆分与祖先噪声注入

## 出处

HuggingFace diffusers 的 `src/diffusers/schedulers/scheduling_euler_ancestral_discrete.py`：`scale_model_input`（`sample / ((sigma**2 + 1) ** 0.5)`）、`step`（\(x_0\) 预测、`sigma_up` / `sigma_down` 拆分、Euler 更新、`randn_tensor` 注入噪声）、`set_timesteps`（σ 栅格与终端 0）。算法来自 Karras et al. 2206.00364 的「ancestral sampling」一节：每步按 \(\sigma_{\text{up}}\) 重新加噪，把 ODE 采样变成有随机性的 SDE 采样。

## 核心与易错点

1. **拆分恒等式**：\(\sigma_{\text{up}}^2+\sigma_{\text{down}}^2=\sigma'^2\)，且 \(\sigma_{\text{up}}^2=\sigma'^2\left(1-\sigma'^2/\sigma^2\right)\)。也就是说「上一级噪声里有多少比例该被重新注入」只取决于两级 σ 的比值；\(\sigma'=\sigma\) 时随机项为 0、\(\sigma'=0\) 时两项都为 0。把 `sigma_down` 误写成 \(\sigma'\)（而不是 \(\sqrt{\sigma'^2-\sigma_{\text{up}}^2}\)）会让每一步多走一截，最后一步也不再精确落到 \(x_0\)。
2. **`dt` 用 `sigma_down` 而不是 `sigma_to`**：`dt = sigma_down - sigma`。确定性部分只负责走到 \(\sigma_{\text{down}}\)，剩下的下降量由噪声项承担。
3. **最后一步是精确的**：\(\sigma'=0\Rightarrow\sigma_{\text{up}}=\sigma_{\text{down}}=0\Rightarrow dt=-\sigma\)，于是 `prev = x + (x-x0)/σ·(-σ) = x0`。这条恒等式是本题最好的自检，也解释了为什么最后一步不需要 `noise`。
4. **`v_prediction` 支照抄了上游的写法**：上游写的是 `model_output * (-σ/√(σ²+1)) + sample/(σ²+1)`，即样本项系数是 \(\alpha^2=1/(\sigma^2+1)\)，而噪声项系数是 \(\sigma_t=\sigma/\sqrt{\sigma^2+1}\)。教科书公式应为 \(x_0=\alpha x-\sigma_t v\)（`LCMScheduler` 与 `DPMSolverMultistepScheduler` 里就是这一版）。可以把它读成「对**已缩放**的样本套用正确公式」：\(\alpha\cdot\text{scaled}-\sigma_t v\)，但同一个 `step` 里的 `epsilon` 支用的却是**未缩放**的 `sample`，两支口径并不一致。本题要求照抄源码，并在对拍里显式验证它与教科书公式不相等、且 \(\sigma\to0\) 时差异按 \(\alpha(1-\alpha)\) 收缩到 0。
5. **`scale_model_input` 只作用于模型输入**：`scaled_input` 是喂给 UNet 的量，`step` 内部的 \(x_0\) 与 Euler 更新全部用原始 `sample`。把两步都换成缩放后的样本会导致结果系统性偏移。
6. **`noise` 是显式入参**：上游用 `randn_tensor` 现采，本题要求调用方传入，才能得到确定性的对拍结果。\(\sigma_{\text{up}}=0\) 时完全不参与计算——实现里若无条件 `+ noise*0` 而 `noise` 含 NaN/Inf，会污染输出。
7. **σ 必须单调不增**：\(\sigma'>\sigma\) 会让 `sigma_from**2 - sigma_to**2` 变负、开方得到 NaN。上游没有校验，本题要求抛 `ValueError`。
8. **数值口径**：上游把 `sample` 升到 float32 再算、最后降回模型 dtype；本题统一 float64 内部计算、返回 float32，因此示例数值与 float64 手算一致。

## 对拍口径

判题用纯 Python + numpy 的 `_oracle` 按同样公式逐元素重做缩放、\(x_0\)、拆分与更新，并把输入先压到 float32 再喂 oracle（与参考解同一批舍入值）：

- **示例值**：三步的 `scaled` / `x0` / `prev` / `sigma_up` / `sigma_down` 六位小数硬编码，最后一步断言 `prev` 与 `x0` 逐位相同；
- **拆分恒等式**：5 级 σ 栅格上验证 \(\sigma_{\text{up}}^2+\sigma_{\text{down}}^2=\sigma'^2\)、\(\sigma_{\text{up}}^2=\sigma'^2(1-\sigma'^2/\sigma^2)\)、\(0\le\sigma_{\text{up}}\le\sigma'\)，以及 \(\sigma'\to\sigma\) 与 \(\sigma'\to0\) 两个极限；
- **终步与静止步**：\(\sigma'=0\) 时 `prev == x0` 且换任意 `noise`（含 1e6）结果不变；\(\sigma'=\sigma\) 时 `dt=0`、`prev == sample`；
- **噪声尺度**：`prev` 与「零噪声版」的差恰好是 `noise·sigma_up`，与 `sigma_down` 无关；
- **`v_prediction` 支**：与上游闭式 \(\alpha\cdot\text{scaled}-\sigma_t v\) 一致，与教科书 \(\alpha x-\sigma_t v\) 明显不等，且 4 档递减的 σ 上差异单调收缩；
- **`epsilon` 往返**：由给定的 \(x_0\) 反算 \(\epsilon\) 再走一步，能还原 \(x_0\) 与解析 `prev`；
- **组合对拍**：2 种形状 × 4 组 σ 栅格 × 每步 × 2 种预测类型，全部与 oracle 一致；
- **非法输入**：18 组（非张量 / 整型 / 形状不符 / sigmas 为 list / 二维 / 长度 1 / 递增 / 含负 / 含 Inf、`step_index` 越界或非整数、未知或 `sample` 预测类型、\(\sigma_{\text{up}}>0\) 缺 `noise`、`noise` 形状不符、当前 σ 为 0）都必须抛 `ValueError`。

共八组断言。

## 复杂度

对样本张量的每个元素做常数次乘加，\(O(B\cdot C\cdot H\cdot W)\) 时间与空间（额外生成缩放输入、\(x_0\) 与下一步样本各一份）。
