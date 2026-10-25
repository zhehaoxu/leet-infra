# 详解：train/eval 模式、推理与 clip 的行为账

## 出处

PyTorch `torch.nn.Module.train/eval` 文档：模式标志只影响 BatchNorm/Dropout 等模块的行为，不触碰 autograd；`nn.BatchNorm1d` 文档写明 train 时用批统计并按时长量更新 \(running\leftarrow(1-m)running+m\,batch\)（批方差为无偏），`num_batches_tracked` 仅在 train 递增。`torch.inference_mode` 文档：模式内产生的 inference tensor 不支持 autograd，模式外对它就地 `requires_grad_(True)` 直接报错——这与 `no_grad`（普通张量、可再开启）形成对照。`clip_grad_norm_` 文档：`norm_type=inf` 时总范数为所有梯度的最大绝对值，且仅当 `total_norm > max_norm` 时全体乘 `max_norm/(total_norm+1e-6)`。

## 核心与易错点

1. **eval() ≠ 关梯度**：`bn.eval()` 后仿射参数仍 `requires_grad=True`，eval 前向输出带 grad_fn。把"推理"写成只调 `eval()` 却不做 `no_grad/inference_mode`，白白记录整张计算图。
2. **BN 的 running 账是无偏 + 指数滑动**：批方差除以 \(N-1\)；两条更新式共享 momentum=0.1。本题三通道两批数据的 \(|b-a|\) 各为 0.75/2.5，所以无偏方差同为 \(0.28125/3.125\)，running_var 三通道同值——这是手算锚点。
3. **Dropout 是倒置缩放**：train 下保留元素乘 \(\frac{1}{1-p}\)，全 1 输入下值域恰为 \(\{0,\frac{1}{0.75}\}\)；eval 下原样返回（恒等）。计数 \(8/20\) 依赖 seed=7 的伯努利流，只能对示例精确、对其它解做不变量校验。
4. **inference tensor 的两道墙**：`.backward()` 抛 "element 0 of tensors does not require grad and does not have a grad_fn"；模式外 `.requires_grad_(True)` 抛 "Setting requires_grad=True on inference tensor outside InferenceMode is not allowed"。`no_grad` 输出没有第二道墙。
5. **clip 是均匀缩放不是截断**：\(-4\) 和 \(-0.5\) 乘同一个 \(coef=1/(4+10^{-6})\)；小梯度也会被缩小（1.0→0.24999994）。+1e-6 兜底使裁剪后最大范数严格小于 max_norm。
6. **eval 前向不动状态**：running/num_batches 全冻结，`state_dict()` 前后 `torch.equal`——这是"eval 污染训练统计"类 bug 的最小可检验版本。

## 对拍口径

solution 走真实框架；`_oracle` 纯 Python：逐通道 \((1-m)rm+m\,\bar{x}\)、无偏 \(\sum(x-\bar{x})^2/(N-1)\) 手算 running 账，eval 输出用 \((x-rm)/\sqrt{rv+10^{-5}}\) 闭式，clip 账用 \(coef\) 逐元素乘——全部与解 `allclose(1e-5,1e-5)`。Dropout 计数无法用纯 Python 复现（torch Philox 流），故示例精确断言 8，另设 \(\{0,4/3\}\) 值域与 \(0<n<20\) 不变量。布尔/计数/total_norm 精确断言。

## 复杂度

\(O(B\cdot C+P)\)，B=2 批 ×3 通道、P=6 个梯度元素，常数级。
