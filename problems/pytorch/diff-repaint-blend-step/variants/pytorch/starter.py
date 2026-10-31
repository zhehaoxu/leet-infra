def solve(sample, model_output, original_image, mask, noise, betas, undo_noise, timestep,
          num_inference_steps, eta=0.0, clip_sample=True):
    """RePaint 的掩码拼接步与 undo_step 前向回跳（diffusers `RePaintScheduler`）。

    语义约定
    --------
    一次调用同时复现上游 `step()`（Algorithm 1 Line 8 的掩码拼接）与 `undo_step()`
    （Algorithm 1 Line 10 的前向回跳）。上游自己 draw 的随机数、`alphas_cumprod` 表
    全部改成显式入参：

    - `sample`：当前样本 x_t（第 0 维为 batch）。
    - `model_output`：网络在当前步的 ε 预测。
    - `original_image`：原图。掩码为 1 的区域最终只由它和本步噪声决定，网络输出碰不到它。
    - `mask`：与 `sample` 同形状，取值在 [0, 1]。**1 = 保留原图**，0 = 要补的区域；
      可以是 0/1 也可以是软掩码。
    - `noise`：本步的**唯一一张**高斯噪声。上游只 draw 一次，同一张既加到已知区
      （`sqrt(1 - ᾱ_prev) · noise`）也加到未知区的 η 项（`std_dev_t · noise`），
      所以这两个系数的相对大小会直接体现在结果里。
    - `betas`：一维 β 表，长度 T。`ᾱ_t = prod_{j<=t} (1 - beta_j)`（前缀积，含 t）。
    - `undo_noise`：形状 `(n,) + sample.shape`，其中 `n = T // num_inference_steps`；
      回跳的第 i 个子步用 `undo_noise[i]`。
    - `timestep`：t，**直接作为 `betas` / `ᾱ` 的下标**（不是 `timesteps` 数组里的位置）。
    - `num_inference_steps`：N。子步长 `n = T // N`，上游用整除，故要求 `T % N == 0`。
    - `eta`：DDIM 式 σ 插值的权重，>= 0；可为 0，也可大于 1。
    - `clip_sample`：是否把 x̂₀ 截到 [-1, 1]。

    公式（中间量在 float64 下算）
    ----------------------------
    `prev = t - n`；`ᾱ_prev = ᾱ[prev]`，当 `prev < 0` 时取 **1.0**（上游的
    `final_alpha_cumprod`，注意不是 `ᾱ[0]`）。记 `ᾱ_t = ᾱ[t]`、`β_prod_t = 1 - ᾱ_t`。

    ```
    x0_hat = (sample - sqrt(β_prod_t) * model_output) / sqrt(ᾱ_t)
           然后（clip_sample 为真时）截到 [-1, 1]
    var    = ((1 - ᾱ_prev) / β_prod_t) * (1 - ᾱ_t / ᾱ_prev)
    std    = eta * sqrt(var)
    add    = std * noise        仅当 t > 0 且 eta > 0，否则整个 η 项为 0
    dir    = sqrt(1 - ᾱ_prev - std**2) * model_output
    unknown = sqrt(ᾱ_prev) * x0_hat + dir + add
    known   = sqrt(ᾱ_prev) * original_image + sqrt(1 - ᾱ_prev) * noise
    prev_sample = mask * known + (1 - mask) * unknown
    ```

    `var` 里的 `1 - ᾱ_prev` 是 `beta_prod_t_prev`；`dir` 的平方根里用的是
    **`ᾱ_prev`**（不是 `ᾱ_t`）再减 `std**2`。

    回跳从 `sample` 出发，逐个子步做前向扩散（**每个子步一个 β，不是一次大跳**）：

    ```
    for i in 0 .. n-1:
        sample_{i+1} = sqrt(1 - betas[t + i]) * sample_i + sqrt(betas[t + i]) * undo_noise[i]
    ```

    返回
    ----
    `(prev_sample, pred_original_sample, undone)`：三个与 `sample` 同形状的 float32
    张量。第二项是（可能已截断的）x̂₀，即上游 `RePaintSchedulerOutput` 的
    `pred_original_sample`。

    非法输入一律抛 `ValueError`
    --------------------------
    非张量；张量少于 1 维；含 NaN/inf；`sample/model_output/original_image/mask/noise`
    形状不一致；`mask` 取值越出 [0, 1]；`betas` 不是一维、含非有限值、有元素越出
    [0, 1)，或前缀积 `ᾱ` 触到 0；`undo_noise` 不是张量或形状不等于 `(n,) + sample.shape`；
    `timestep` / `num_inference_steps` 不是 Python int（`bool` 非法）或越界
    （`num_inference_steps >= 1`、`0 <= timestep`、`timestep + n <= T`）；
    `T % num_inference_steps != 0`；`eta` 不是有限非负实数；`clip_sample` 不是 bool；
    `eta` 大到 `1 - ᾱ_prev - std**2` 出现负值。

    禁止 `import diffusers`。
    """
    raise NotImplementedError("请实现 solve()")
