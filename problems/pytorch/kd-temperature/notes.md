# 详解：知识蒸馏温度损失

## 出处
Hinton, Vinyals, Dean, *Distilling the Knowledge in a Neural Network*（NIPS 2014 Deep Learning Workshop，arXiv:1503.02531）。软目标损失定义为 \(T^2\,\mathrm{KL}\big(\mathrm{softmax}(z_t/T)\,\|\,\mathrm{softmax}(z_s/T)\big)\)；乘 \(T^2\) 是因为 softmax 温度求导会带 \(1/T\)，需保持硬/软两路损失的梯度量级可比。

## 核心与易错点
1. **KL 方向是 teacher‖student**：以教师分布为「参考测度」，\(p_t\) 作权重，写成 \(\sum p_t(\log p_t-\log p_s)\)。方向反了数值不同。
2. **先除以 T 再 log_softmax**：\(\log\mathrm{softmax}(z/T)\) 数值稳定，避免显式 `softmax` 后取 `log`。
3. **\(T^2\) 在归约前**：逐样本 KL 乘 \(T^2\) 后再 mean/sum/none，而不是先归约再乘。
4. **退化锚点**：\(T=1\) 时 \(1^2\cdot\mathrm{KL}(\mathrm{softmax}(z_t)\|\mathrm{softmax}(z_s))\)，用它对拍普通 softmax-KL。
5. **reduction 三态**：`none` 返回 \((N,)\) 向量，非法串抛 `ValueError`。

## 对拍口径
参考解 `log_softmax(z/T)→ kl_rows=(p_t*(tt-ts)).sum(-1)→ loss=T*T*kl_rows → reduction`。判题 `_oracle` 纯 Python：先手算每行 \(z/T\) 的 softmax（减最大值），再按定义 \(\sum p_t\ln(p_t/p_s)\) 逐样本累加、乘 \(T^2\)、按 reduction 归约，结构不同、数值等价。另设「T=1 等于普通 softmax-KL」「师生相同→0」「example 数值」「非法 reduction 抛错」断言。

## 复杂度
\(O(N C)\)。
