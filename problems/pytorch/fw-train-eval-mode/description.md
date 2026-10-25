# train/eval 模式、推理与 clip 的行为账

## 题目描述

`model.eval()` 只切换 BatchNorm/Dropout 这类模式敏感算子的行为，不关 autograd；`torch.inference_mode()` 则连梯度记录都禁止，产出的推理张量甚至不允许事后 `requires_grad_(True)`。梯度裁剪 `clip_grad_norm_(..., norm_type=inf)` 用的是最大绝对值做总范数，系数对所有梯度统一相乘。本题按五个阶段把这些模式行为全部做成可逐位检验的账目。

## 规则

实现 `solve()`，按 starter docstring 依次跑五个阶段并返回行为账 dict：

1. **阶段 A**：默认参数 `nn.BatchNorm1d(3)`（momentum=0.1、eps=1e-5）两次 train 前向，记录 running_mean/running_var（无偏方差、\(running\leftarrow0.9\,running+0.1\,batch\)）与 `num_batches_tracked`。
2. **阶段 B**：`bn.eval()` 后对 `xe` 前向，验证 state_dict 逐张量不动，并用 running 统计手算归一化输出。
3. **阶段 C**：`torch.manual_seed(7)`、`nn.Dropout(0.25)` 对全 1 输入：train 下值域恰为 \(\{0,\ \frac{1}{0.75}\}\)（倒置缩放），eval 下恒等。
4. **阶段 D**：比较三种"免梯度"途径：eval 前向输出仍 `requires_grad`；`inference_mode` 输出不可 backward、不可就地开启；`no_grad` 输出可开启。
5. **阶段 E**：给两个 Parameter 手工置 grad 后 `clip_grad_norm_([w1, w2], max_norm=1.0, norm_type=inf)`：总范数 = 最大绝对值 \(=4.0\)，裁剪系数 \(coef=1/(4+10^{-6})\) 作用于每个元素。

## 示例

```
输入：无
输出：
{
 "running_mean_after2": [0.058750003576278687, 0.13124999403953552, 0.2537499964237213],
 "running_var_after2": [1.1478124856948853, 1.1478124856948853, 1.1478124856948853],
 "num_batches_train": 2,
 "eval_state_unchanged": true,
 "eval_out_requires_grad": true,
 "eval_out_flat": [0.41185787320137024, -1.055896282196045, 1.6299304962158203,
                   0.8785523176193237, 0.11083994060754776, -0.7035419344902039],
 "dropout_train_unique": [0.0, 1.3333333730697632],
 "n_dropout_zeros": 8,
 "dropout_eval_identity": true,
 "inference_requires_grad": false,
 "inference_backward_raises": true,
 "inference_enable_raises": true,
 "no_grad_enable_ok": true,
 "total_norm_inf": 4.0,
 "grads_after_flat": [0.7499998211860657, -0.9999997615814209, 0.0,
                      0.49999988079071045, 0.24999994039535522, -0.12499997019767761],
 "clip_scaled_uniformly": true
}
```

三个通道的无偏批方差都恰好是 \((b-a)^2/2\in\{0.28125\,(x_1),\ 3.125\,(x_2)\}\)，所以 running_var 三通道同值。

## 约束

- running 统计、eval 输出、裁剪后梯度用 `atol=1e-5, rtol=1e-5`；布尔、`num_batches_train`、`n_dropout_zeros`、`total_norm_inf` 精确一致。
- 阶段 C 只允许依赖 `manual_seed(7)` 的确定性；不许改用固定 mask。
- 阶段 D 的三个 raises/ok 标志按捕获 `RuntimeError` 与否记录，不许手写常量绕过。
