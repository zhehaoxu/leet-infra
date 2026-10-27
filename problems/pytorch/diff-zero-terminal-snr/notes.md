# 详解：zero-terminal-SNR 重标定与 log-SNR 参数化

## 出处

Lin et al., *Common Diffusion Noise Schedules and Sample Steps are Flawed*（arXiv:2305.08891）§3.1：指出训练计划末端 SNR 不为 0、采样末步仍留噪声是一致性问题的根源，给出重标定公式并要求配 v-prediction；diffusers 的 `rescale_betas_zero_snr=True` 与 `prediction_type="v_prediction"` 即这两步。

## 核心与易错点

1. **重标定只缩放 \(1-\bar\alpha\)**：\(\bar\alpha'=(\bar\alpha-\bar\alpha_T)/(1-\bar\alpha_T)\) 与「把 \(\bar\alpha\) 按比例缩放」完全不同；前者保 \(\bar\alpha'_0=1\)，后者不保。
2. **末端必须是精确 0**：判题用 `float(alpha_bars[-1]) == 0.0` 断言。用 `1 - 1e-7` 近似会在下一步 \(\log(0)\) 之外的地方出问题：\(\lambda_T\) 需要是真 \(-\infty\)。
3. **±inf 不能靠一次统一公式算**：\(\lambda=\log(\bar\alpha/(1-\bar\alpha))\) 在两端分别是 \(-\infty\)/\(+\infty\)，实现必须分支，并在出口把它们写进 float32 张量（`torch.tensor(float('-inf'))` 是合法值）。
4. **为什么必须 v-prediction**：末端 \(\alpha=0\)，\(\epsilon\) 参数化反解 \(x_0=(x_t-\sigma\epsilon)/\alpha\) 直接除零；v 参数化 \(x_0=\alpha x_t-\sigma v\) 在 \(\alpha=0\) 时仍有定义（退化为 \(-\sigma v\)）。这就是 \(\theta_T=\pi/2\)、\(v=-x_0\) 的含义。
5. **均匀 β 有闭式**：\(\bar\alpha'_t=((1-\beta)^{t+1}-(1-\beta)^n)/(1-(1-\beta)^n)\)，可直接验算重标定没有把比值改坏。

## 对拍口径

参考解先 `cumprod` 再重标定（float64），逐项用 `math` 算 \(\lambda\) 与 \(\theta\)。判题 `_oracle` 用 python 循环滚累积乘积（不调 `cumprod`）并逐个重算，`atan` 版代替 `atan2` 版；另加五组结构断言：末端零 SNR 三项（ᾱ=0、λ=-inf、θ=90°）、首项 \(\beta_0=0\) 的三项（ᾱ=1、λ=+inf、θ=0）、\(\cos\theta/\sin\theta\) 与 \(\alpha/\sigma\) 自洽、三项单调性、\(\theta=\arctan(e^{-\lambda/2})\) 与均匀 β 闭式。阈值 `allclose(atol=1e-6)`。

## 复杂度

\(O(T)\)：一次累积乘积加两次逐项超越函数。
