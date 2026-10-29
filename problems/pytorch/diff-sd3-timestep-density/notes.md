# 解题说明 · SD3 的时间步采样密度

## 出处

- `huggingface/diffusers` 的 `src/diffusers/trainers/training_utils.py`：`compute_density_for_timestep_sampling(weighting_scheme, batch_size, logit_mean, logit_std, mode_scale, ...)`。三个分支就是 `logit_normal`（`torch.normal` 后套 `sigmoid`）、`mode`（均匀抽样后做余弦变换）、以及其它/兜底（`torch.rand`）。本题只保留纯函数部分，把抽样结果作为入参。
- Esser et al., *Scaling Rectified Flow Transformers for High-Resolution Image Synthesis*（SD3，arXiv:2403.03206）给出该密度设计与 `mode` 采样；diffusers 的实现由 PR #8528 引入（源码 docstring 注明 contributed by Rafie Walker）。
- 返回的是 \([0,1]\) 上的**连续位置**，训练脚本再乘 `num_train_timesteps` 取整；它不是一个整数时间步。

## 核心与易错点

**`logit_normal` 抽的是正态、不是均匀。** 名字里的 logit 来自「对正态样本过 sigmoid 得到 logit-normal 分布」，而不是「对 logit 做均匀采样」。把输入当成 \(U(0,1)\) 直接套 sigmoid，分布形状完全不同——这一步在本实现里通过 `torch.normal` 完成，本题把 \(z\) 交给调用方。

**`mode` 的符号与括号结构。** 公式是 `1 - u - s * (cos²(πu/2) - 1 + u)`：先 `1-u`，再减去 `s` 乘上整个括号，括号里是 `cos² - 1 + u`。常见的三种写错方式：把 `1-u` 写成 `u`；把 `cos²` 写成 `cos`；漏掉括号里的 `-1+u`（只保留 `cos²`）。`u=0.5` 恰好给出 0.5 是校验括号是否写对的快照。

**`mode_scale=0` 不是恒等。** 它退化成 `1-u`（方向翻转的均匀），不是 `u`；只有 `uniform` 才是恒等。题目把这条作为显式检查。

**端点与对称性。** 任意 \(s\ge0\) 都有 \(v(0)=1\)、\(v(1)=0\)、\(v(0.5)=0.5\)，且 \(v(u)+v(1-u)=1\)。前三条来自余弦项的端点值，第四条来自 \(\cos^2(\pi u/2)+\sin^2(\pi u/2)=1\)。这些性质与 \(s\) 无关，适合当性质测试。

**`logit_std=0` 的退化。** 此时整批都等于 \(\sigma(\mu)\)，不再是严格单调的；测试里单独处理，不能和「随 z 严格递增」的性质混在一起断言。

**兜底行为的差异。** 源实现对任何未知字符串都静默走 uniform 分支；本题把范围收紧为「只接受三个合法名字，其余抛 `ValueError`」，以免拼写错误被悄悄吞掉。这是本题对源码的有意收窄，notes 里明示。

**精度。** 输入转 float64 计算、出口转 float32；`cos²`、`exp` 的连乘在 float32 下与 float64 oracle 相比会在 1e-7 量级漂移，性质测试留 1e-6 容差。

## 与其他题的分工

- `diff-rf-timestep-shift`：给定时间后做单参数 shift（把栅格推向噪声侧）；本题解决「时间从哪里来」。
- `diff-zero-terminal-snr`：改的是噪声调度的 \(\bar\alpha_t\) 端点；本题只改抽样密度，不动调度。
- `diff-ddim`、`diff-ddpm-forward`：采样器/前向过程；本题是训练侧的抽样策略。
- `diff-min-snr-weight`：同一训练脚本里紧邻的另一件账——抽完时间步后按 SNR 给损失加权；两题合起来覆盖 SD3 训练的时间步管线。
