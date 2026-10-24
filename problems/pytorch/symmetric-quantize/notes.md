# 对称量化 详解

## 量化三元组

任何均匀量化由三个量描述：

- **scale** $s$：每个整数格点代表的浮点步长。
- **zero_point** $z$：浮点 0 对应的整数格点。对称量化 $z=0$。
- **范围** $[q_{\min},q_{\max}]$：int8 对称常用 $[-127,127]$（留 -128 给非对称/饱和余量）。

$$ \hat x = (q - z)\,s,\quad q=\operatorname{clamp}(\operatorname{round}(x/s),\,q_{\min},\,q_{\max}) $$

对称版 $z=0$，$s=\text{absmax}/127$。

## 为什么 absmax/127 而不是 /128

int8 对称取 $[-127,127]$（正端只到 127），让 $\pm\text{absmax}$ 都精确落在格点边界、关于 0 对称。用 128 会让 $-\text{absmax}$ 映射到 -128 而 $+\text{absmax}$ 只能到 127，破坏对称。很多框架（含 PyTorch `quantize_per_tensor` 的对称口径）就用 127。

## fake-quant 与 STE

推理要真存 int8；训练时则用 **fake-quantize**：量化→反量化往返，值仍是浮点但带上了量化噪声，模型「提前适应」低精度。round 的梯度处处为 0，故用 **直通估计（STE）**：前向 round、反向当 round 是恒等。本题只做前向数值（判题不查梯度），但概念上 fake-quant 就是 STE 的前向部分。

## 误差

往返误差 $|\hat x - x|\le s/2$（就近取整到半格）。absmax 越大、$s$ 越大、噪声越大——离群值会抬高 absmax 从而「稀释」精度，这正是 per-channel / 非对称 / clipping 要缓解的。

## 坑

- **除零**：全零张量 absmax=0，$s=0$ → $x/s$ NaN，需特判原样返回。
- **round 口径**：用 `torch.round`（银行家舍入，0.5→最近偶数），与硬件/`fake_quantize` 一致；别用 floor(x+0.5)。
- clamp 必要：数值误差可能让某个 $|x/s|$ 微超 127。

## 关联

asymmetric-quantize 是其 $z\ne0$ 版（用 min/max 而非 absmax）；hardsigmoid/relu6 的「截断」思想与 clamp 相通；quantization 家族是部署侧省内存/提速的关键。
