# DDIM 逆向一步：把样本推回更噪的一级

## 题目描述

DDIM 采样从纯噪声一路走到干净样本；把它反过来跑，就从干净样本一路走回噪声，这就是 DDIM 逆向编码（inversion），常用于「先编码成噪声、再改写噪声、再解码」的图像编辑。逆向的每一步都要求模型给出比当前样本**更噪**一级的样本，因此除了与采样相同的「预测 x0 + 方向项」外，还要处理一个不对称之处：输出所在的噪声级是当前 `timestep`，而用来预测 x0 的噪声级要再往回退一个步长，退到 0 以下时用一个可配置的初始 alpha 兜底。

本题只做逆向的**单步**：给定累积 alpha 调度、当前时间步、样本与模型输出，算出这一步的 `prev_sample` 和预测出的 `pred_original_sample`。

## 规则

记调度长度为 \(T\)、推理步数为 \(N\)，步长比例为 \(r = \lfloor T/N \rfloor\)（整数除法）。

1. **两个噪声级**：输出级就是 `timestep`；预测级为 \(t=\min(\texttt{timestep}-r,\;T-1)\)。于是
   \[
   \bar\alpha_t=\begin{cases}\bar\alpha[\texttt{timestep}],&t\ge 0\\ \text{initial},&t<0\end{cases}
   \qquad
   \bar\alpha_{\text{prev}}=\bar\alpha[\texttt{timestep}]
   \]
   其中 `initial` 在 `set_alpha_to_one=True`（默认）时取 \(1.0\)，否则取 \(\bar\alpha[0]\)。
2. **按预测类型拆出 x0 与噪声**（\(\beta_t=1-\bar\alpha_t\)）：

   ```
   epsilon:      x0 = (sample - sqrt(beta_t) * model_output) / sqrt(alpha_t)
                 eps = model_output
   sample:       x0 = model_output
                 eps = (sample - sqrt(alpha_t) * x0) / sqrt(beta_t)
   v_prediction: x0 = sqrt(alpha_t) * sample - sqrt(beta_t) * model_output
                 eps = sqrt(alpha_t) * model_output + sqrt(beta_t) * sample
   ```
3. **可选截断**：`clip_sample=True` 时把 x0 夹到 \([-c, c]\)，\(c\) 为 `clip_sample_range`。
4. **合成更噪一级的样本**：
   \[
   \texttt{prev\_sample}=\sqrt{\bar\alpha_{\text{prev}}}\,x_0+\sqrt{1-\bar\alpha_{\text{prev}}}\,\epsilon
   \]

返回 `(prev_sample, pred_original_sample)`：两个与 `sample` 同形状的 float32 张量。注意返回的是**截断后**的 x0。

## 示例

`alphas_cumprod = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]`（\(T=10\)）、`num_inference_steps = 10`（故 \(r=1\)）、`timestep = 5`、`sample = 2.0`、`model_output = 0.5`、`prediction_type = "epsilon"`：

```
预测级 t = 4        alpha_prod_t = 0.5       beta_prod_t = 0.5
输出级 timestep = 5 alpha_prod_t_prev = 0.4

x0  = (2.0 - sqrt(0.5) * 0.5) / sqrt(0.5) = 2.328427
eps = 0.5
```

- 不截断时：`prev_sample = sqrt(0.4) * 2.328427 + sqrt(0.6) * 0.5 = 1.859925`；
- 默认 `clip_sample=True`、`clip_sample_range=1.0` 时：x0 被夹到 1.0，`prev_sample = sqrt(0.4) * 1.0 + sqrt(0.6) * 0.5 = 1.019754`。

若 `timestep = 0`、`sample = 1.0`、`model_output = 0.2`，则预测级 \(t=-1\)、\(\bar\alpha_t\) 取初始值：`set_alpha_to_one=True` 给出 `(1.011929, 1.0)`，改为 `False`（用 \(\bar\alpha[0]=0.9\)）给出 `(1.0, 0.987426)`。

## 约束

- `alphas_cumprod` 必须是一维浮点张量（float32 / float64），非空、不含 nan / inf，且每个元素落在 \((0,1]\) 内；
- `timestep` 是 Python 整数（不接受布尔值），取值 \([0,T)\)——它会被直接用来索引 `alphas_cumprod`，也就是输出所在的噪声级；
- `num_inference_steps` 是整数，满足 \(1\le N\le T\)；
- `sample` 与 `model_output` 必须是同形状、非空、不含 nan / inf 的浮点张量；
- `prediction_type` 只能取 `"epsilon"` / `"sample"` / `"v_prediction"`；
- `clip_sample`、`set_alpha_to_one` 必须是布尔值；`clip_sample_range` 必须是正的有限实数；
- 违反以上任一条抛 `ValueError`；`"sample"` 预测类型若遇到 \(\beta_t\le 0\)（即 \(\bar\alpha_t=1\)，只可能来自 `timestep` 退到 0 以下且 `set_alpha_to_one=True`）也抛 `ValueError`；
- 性质：无截断时，用前向加噪得到的 \((\bar\alpha_t,\,x_0,\,\epsilon)\) 反推，`prev_sample` 精确等于前向过程在 `timestep` 这一级的样本；三种预测类型在同一组 \((x_0,\epsilon)\) 下给出相同的 `prev_sample`。
