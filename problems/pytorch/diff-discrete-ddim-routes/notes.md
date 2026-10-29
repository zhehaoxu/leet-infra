# 详解：D3PM 均匀腐蚀的后验三路路由

## 出处

*Structured Denoising Diffusion Models in Discrete State-Spaces*（D3PM，arXiv:2107.03006）的均匀腐蚀核。实现口径对齐 HuggingFace diffusers 的 `src/diffusers/schedulers/scheduling_discrete_ddim.py`：`step` 里的 `_sample_from_logits` 取贪心/温度采样得到 `sampled_tokens`，随后按 `clean_mass / stay_mass / noise_mass` 堆成三元组、除以自身之和，再 `multinomial` 选路径；`_to_loo_logits` 实现 leave-one-out 去噪器修正。本题只取档位为 `temperature=0`（贪心）时的确定性部分，随机采样不进判题。

## 核心与易错点

1. **归一化不是摆设**：三条路径质量在 `same=False` 的位置和只有 \((1-ab)/K\)，直接当概率用会低估；源码就是除以自身之和，照抄即可。
2. **`a` 与 `b` 的分子分母**：`a = α_t/α_s` 是「从 s 存活到 t」的概率，`b = α_s`。把 `a` 写成 `α_s/α_t` 会让最后一步（\(\alpha_s=1\)、\(\alpha_t=1-1/N\)）不再确定性提交。
3. **最后一步必须确定性提交**：\(\alpha_s=1\Rightarrow b=1\Rightarrow\) stay 与 noise 都含 \((1-b)\) 因子而归零，只剩 clean，归一化后是 one-hot 的 `[1,0,0]`。这是离散 DDIM 与连续 DDIM 共享的「终端 SNR 无限」性质。
4. **`same` 是逐位置的指示量**：命中 \(x_0\) 的位置 clean 多拿一份 \(a\,b\)，未命中的位置没有。用全局标量或按整 batch 统计都会错。
5. **LOO 的平移量与时间有关**：\(\delta=\operatorname{log1p}(K\alpha_t/(1-\alpha_t))\)，\(\alpha_t\to1\) 时发散；源码用 `log1p` 保证小量精度，并在 \(\alpha_t\le 0\) 或 \(\ge 1\) 时直接跳过。本题的 `step_index` 范围保证 \(\alpha_t<1\)，但 `step_index=0` 时 \(\alpha_t=0\) 必须跳过。
6. **LOO 只压「已观测 token」那一个 logit**：它修正的是「模型看到自己刚写下的 token 就自我强化」的偏差，其它位置不动；用 `scatter_add`（而不是覆盖）才是对的加法方向（源码用 `scatter_add` 加一个负值）。
7. **argmax 的取值语义**：`torch.argmax` 在并列时返回第一个最大值的下标；oracle 里手写的 `_argmax` 必须用严格大于（`>`）才和它一致。

## 对拍口径

参考解 float64 计算，`route_probs` 返回 float32、`pred_tokens` 返回 int64。判题 `_oracle` 不用任何张量算子：alpha 账、LOO 平移量、argmax、三路质量、归一化全部用 python 标量逐位置重算。七组断言：示例三档 + LOO 前后对照、最后一步的 one-hot 提交（多种 N）、第一步的 \([\alpha_s,0,1-\alpha_s]\) 结构、未归一化质量之和等于 \((1-ab)/K\)（`same=False` 处）、LOO 的平移量与 `step_index=0` 不修正、九组 (K, L, N, step, loo) 的逐位对拍、非法输入。`pred_tokens` 用 `torch.equal` 精确比对，`route_probs` 用 `atol=1e-6`。

## 复杂度

逐位置 \(O(K)\) 取 argmax 加 \(O(1)\) 质量计算，总计 \(O(BLK)\)，无状态、无额外缓冲。
