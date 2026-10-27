# 详解：CFG 的 rescale 与 guidance interval

## 出处

Lin et al., *Common Diffusion Noise Schedules and Sample Steps are Flawed*（arXiv:2305.08891）§3.2 的 guidance rescale：按 \(\operatorname{std}(\epsilon_c)/\operatorname{std}(\hat\epsilon)\) 缩放，解决高 \(w\) 下的过曝；Kynkäänniemi et al., *Applying Guidance in a Limited Interval Improves Sample and Distribution Quality*（arXiv:2211.11559）给出 guidance interval。相关但不同的手段是 Imagen 的 dynamic thresholding（对 \(x_0\) 做阈值裁剪），不要混用。

## 核心与易错点

1. **std 是「每样本一个标量」**：要在 batch 内逐样本对（通道×空间）做归约，dims 不能含第 0 维。把整个 batch 一起求 std 会让不同样本互相干扰，判题里「缩放必须是每样本一个常数」直接失败。
2. **总体标准差而不是无偏**：口径是 `unbiased=False`。用默认的 \(N-1\) 会带来 \(\sqrt{N/(N-1)}\) 的系统性偏差，逐样本都偏一点。
3. **只保幅度、不保均值**：\(\hat\epsilon\cdot r\) 与 \(\hat\epsilon\) 共享均值方向但均值也被乘了 \(r\)；写成「减均值再缩放再加回」是另一种算法，会让 \(\varphi=0.5\) 的混合断言失败。
4. **区间是闭区间**：\(t=t_{\text{low}}\) 与 \(t=t_{\text{high}}\) 都要引导；判题专门在两端点上取值。区间外必须逐位等于 `eps_cond`（不是「近似等于」）。
5. **std 为 0 要报错**：两支都是常数时 rescale 会变成 0/0 或 inf，必须抛 `ValueError`；这是采样器里真实存在的「degenerate batch」。
6. **\(\varphi\) 混合作用在缩放后**：先算 \(\hat\epsilon\cdot r\) 再与 \(\hat\epsilon\) 线性插值，等价于 \(\hat\epsilon\cdot((1-\varphi)+\varphi r)\)；两种写法数值上一样，但只有前者能表达「部分纠正」。

## 对拍口径

参考解向量化（float64）出结果，`torch.std(..., unbiased=False)` 归约。判题 `_oracle` 逐样本把张量展平成 python 列表、自己算均值与均方偏差（不使用 `torch.std`），另加四组结构断言：\(w=1\) 与区间外逐位恒等、区间端点生效、\(\varphi=1\) 的 std 匹配与 \(\varphi=0.5\) 的线性混合、显式 `dims` 只按最后一维统计。阈值 `allclose(atol=1e-5)`。

## 复杂度

逐元素 \(O(N)\)，外加一次逐样本的方差归约（同样 \(O(N)\)）。
