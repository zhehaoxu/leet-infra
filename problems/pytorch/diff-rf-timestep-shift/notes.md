# 详解：Rectified Flow 的 timestep shift 映射

## 出处

Esser et al., *Scaling Rectified Flow Transformers for High-Resolution Image Synthesis*（SD3，arXiv:2403.03206）§3 给出 \(t'=s\,t/(1+(s-1)t)\) 与按分辨率线性插值 \(\mu\) 的采样账；Flux 的 `compute_empirical_mu`（`base_seq_len=256`、`max_seq_len=4096`、`base_shift=0.5`、`max_shift=1.15`）是同一套常数。

## 核心与易错点

1. **是 logit 域平移，不是乘 t**：\(t'=\sigma(\operatorname{logit}(t)+\mu)\)。直接写 \(t\cdot s\) 会越界且端点漂移；写成 \(\operatorname{logit}\) 版本后端点自动固定（\(0\to0\)、\(1\to1\)）。
2. **\(\mu\) 与 \(s\) 的区别**：\(\mu\) 可正可负、以「扰动量」为单位；\(s=e^{\mu}>0\) 是乘性尺度。判别公式里出现的到底是哪一个（接口同时支持、但必须恰给一个）。
3. **插值是线性外推**：\(\mu=m\cdot\text{seq\_len}+b\) 且不裁到 `[base_shift, max_shift]`，所以 64 token 会得到比 `base_shift` 更小的 \(\mu\)。若实现里加 `clamp`，短序列的断言会失败。
4. **可加性是最好的单元测试**：\(\sigma(\sigma(\operatorname{logit}t+\mu_1)+\mu_2)=\sigma(\operatorname{logit}t+\mu_1+\mu_2)\)，故「两次 shift = 一次乘积极 shift」。
5. **逆映射**：\(t=t'/(s-(s-1)t')\)，与正向同形；采样器做反演（image-to-image、编辑）时要能原样回来。

## 对拍口径

参考解把栅格整体转 float64 后一次算出。判题 `_oracle` 走另一条式子：逐元素 `sigmoid(log(t/(1-t)) + mu)`（端点单独返回 0/1，避开 logit 的 ±inf），与参考解的有理式不同源；另加四组结构断言：端点与单调性、恒等（\(s=1\)）与可加、`seq_len` 插值的三点（base / max / 中点）与单调、逆映射往返。阈值 `allclose(atol=1e-6)`。

## 复杂度

逐元素 \(O(N)\)，只多一次 `exp` 与两次除法。
