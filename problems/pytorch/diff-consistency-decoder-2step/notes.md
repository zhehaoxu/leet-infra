# 题解笔记：Consistency Decoder 的两步调度表

## 出处

- HuggingFace diffusers v0.42.1 `src/diffusers/schedulers/scheduling_consistency_decoder.py`：`betas_for_alpha_bar` 在第 13 行标注「Copied from diffusers.schedulers.scheduling_ddpm.betas_for_alpha_bar」，函数体 14–63（cosine 40–41、laplace 45–48、exp 52–53、`min(1 - ab(t2)/ab(t1), max_beta)` 在 62）；`ConsistencyDecoderScheduler.__init__` 在 100–119（`sigmas = sqrt(1/ab - 1)` 在 113，`sqrt_recip_alphas_cumprod` 在 115，三组系数 117–119）；`set_timesteps` 在 121–134（硬编码 `[1008, 512]` 在 129，`num_inference_steps != 2` 直接抛错在 126）；`init_noise_sigma` 在 136–146；`scale_model_input` 在 148–163；`step` 在 165–212（`x_0` 196，`timestep_idx` 198，末步判据 200–201，重加噪 204–207）。
- 同一份 `betas_for_alpha_bar` 也出现在 `scheduling_ddpm.py`，本题的三种变换与 DDPM 的 cosine 完全同源，`laplace` 是该函数在 v0.42.1 里多出来的分支。

## 核心与易错点

- **`c_skip`/`c_in` 用的是 \(\bar\alpha^{-1/2}\) 而不是 \(\sqrt{\bar\alpha}\)**。源码第 115 行先把 `alphas_cumprod` 取倒数再开方，命名 `sqrt_recip_alphas_cumprod` 就是在提醒这一点。在 \(t=1008\) 处 \(\bar\alpha=5.21\times10^{-4}\)，两种写法差 \(1/\bar\alpha\approx 1919\) 倍，`scaled_first` 会整体缩小三个数量级；而 \(t=512\) 处只差约 2 倍，很容易在小 \(T\) 的手算里被当成舍入误差放过去。`check_three_transforms` 用独立推导的 \(\sqrt{1+\sigma^2}\)（\(\bar\alpha^{-1/2}=\sqrt{1+\sigma^2}\) 是恒等式，但推导路径不同）来卡这一条。
- **重加噪取的是下一个时间点的系数**。第 204–207 行写的是 `self.sqrt_alphas_cumprod[self.timesteps[timestep_idx + 1]]`，即先在 `self.timesteps` 里找到当前步的位置，再取**后一个元素**的值去索引长度为 \(T\) 的数组。因此 \(x_{\text{second}}\) 的 \(\sqrt{\bar\alpha}\) 来自 \(t_1=512\)（0.701656），而不是当前步 \(t_0=1008\)（0.022827）。`check_renoise_uses_next_timestep` 把这两组数值各自能唯一区分的场景做了一遍。
- **末步不加噪**。第 200–201 行的 `if timestep_idx == len(self.timesteps) - 1: prev_sample = x_0` 使 `noise` 只进入 \(x_{\text{second}}\) 一次；\(x_{\text{final}}\) 对 `noise` 的偏导只剩经由 `x_second` 的那一条链。`check_last_step_is_plain_denoise` 用 `f_second=0, noise=0` 与 `noise=1` 两次调用相减，得到 `c_skip[t1]*sqrt(1-ab[t1])` 这条唯一通路。
- **`timestep` 是数组下标，不是步序号**。`scale_model_input`（163 行）和 `step`（196 行）都用 `self.c_*(timestep)` 直接索引长度为 \(T\) 的系数表，所以题面把 \((1008, 512)\) 作为**值**入参并要求它们落在 \([0,T)\) 且严格递减；把 \(t_1\) 当成「第二步」写成索引 1 会同时改坏 `scaled_second` 与 `x_final`。
- **`max_beta` 只在尾部起作用**。cosine 在 \(T=1024\) 时 `beta_0 = 4.026\times10^{-5}`，直到 `beta_1023` 的原始值 \(1.0\) 才触顶被截成 \(0.999\)；laplace 的 \(\beta\) 呈 U 形（\(T=8\)：0.183、0.063、0.044、0.034、0.037、0.055、0.103、0.945），exp 是常数 \(e^{-12/T}\)（\(T=8\) 时 \(0.77687\)）。这三条都在 `check_max_beta_clamp` 里显式测过。
- **调度表退化要拦**。`max_beta=1.0` 配 cosine 时最后一步的 \(\beta\) 恰好等于 1，\(\bar\alpha\) 归零，第 113 行的 `1/ab - 1` 直接除零（上游拿到 `inf`，本题按 Python 语义变成 `ZeroDivisionError`），所以题目改成提前抛 `ValueError`；同参数下 exp 变换的首步 \(\beta=e^{-12/T}\) 远小于 1，不会退化。

## 与上游的差异

- 上游在构造期就固化了 `betas_for_alpha_bar(num_train_timesteps)` 的默认参数（`max_beta=0.999`、`cosine`），`set_timesteps` 拒收任何 \(N\neq2\)；本题把变换名、`max_beta` 与两个时间点都抬成入参，好让三种变换和截断行为都能被单独测到，但保留「两步」这个结构。
- 上游 `betas_for_alpha_bar` 返回 float32 张量，`cumprod` 也在 float32 上进行；本题要求中间量走 float64。与逐行转写的上游实现对比，最差相对偏差 \(2.3\times10^{-6}\)，正是 float32 累积舍入的量级。
- 上游 `step` 自己调 `randn_tensor` 抽噪声；本题把 `noise` 作为显式入参，评测不依赖随机数生成器，也不需要 `generator`。
- 去掉了 `SchedulerMixin / ConfigMixin / register_to_config`、`ConsistencyDecoderSchedulerOutput` 和 `return_dict` 分支，返回值改为四元组，把 `scale_model_input` 的两处结果一并交出来，便于分别定位错误。

## 与其他题的分工

- `diff-noise-schedule` 负责 \(\beta/\bar\alpha/\sigma\) 三者的通用互转；本题的新内容是 `betas_for_alpha_bar` 的三种变换、`max_beta` 截断，以及「系数按时间点的**值**索引」。
- `diff-sigma-grid-construct` / `diff-rf-timestep-shift` 研究的是推理期重新铺一张 \(\sigma\) 栅格；本题的栅格来自训练期的累积乘积，两者构造路径不同。
- `diff-edm-preconditioning` 用 \((c_{\text{skip}}, c_{\text{out}}, c_{\text{in}})\) 这套名字，但公式是 EDM 的 \(\sigma_d/(\sigma^2+\sigma_d^2)\) 形式且 \(\sigma\) 由外部给定；本题的 \(\bar\alpha^{-1/2}\) 因子来自 DDPM 侧的反归一化，是同款命名的另一条血统。
- `diff-ddpm-step` 走 \(\epsilon\) 预测 + 后验均值的标准更新；本题网络直接出 \(x_0\)，去噪后是「跳到下一个时间点重加噪」，与 `diff-lcm-boundary-step` 的两步边界裁剪最接近，区别在 LCM 仍按 \(\epsilon\) 预测反解 \(x_0\)，而这里根本不存在噪声预测。
