# 详解：MAMBO-G 的幅值自适应引导

## 出处

*Magnitude-Aware Mitigation for Boosted Guidance*（arXiv:2508.03442）。实现口径对齐 HuggingFace diffusers 的 `src/diffusers/guiders/magnitude_aware_guidance.py`：`mambo_guidance`（L141–159）就是全部内容——先算 `dim = list(range(1, len(pred_cond.shape)))`，再求 `torch.norm(diff, dim=dim, keepdim=True) / torch.norm(pred_uncond, dim=dim, keepdim=True)`，最后 `guidance_scale_final = 1.0 + (guidance_scale - 1.0) * torch.exp(-alpha * ratio)`。同一函数的 `use_original_formulation` 分支把基准换成 `pred_cond`、系数换成 `guidance_scale * exp(-alpha * ratio)`；本题只取 diffusers 原生口径（基准 \(\epsilon_u\)）。

## 核心与易错点

1. **范数是逐样本标量，不是逐元素**：归约维度是 `range(1, ndim)`。写成逐元素比值 \(|\delta_i|/|u_i|\) 会让每个元素拿到不同的 \(w'\)，等于换了一个算子。
2. **衰减作用在「有效尺度」上，不是乘在结果上**：正确写法是 `out = u + w'·δ`，其中 \(w'\) 已经含了 \(1\) 这个偏移。写成 `out = w'·(u + w·δ)` 会把无条件预测也一起缩放。
3. **\(w'=1+(w-1)e^{-\lambda r}\) 的常数 1 不能省**：省掉就变成 \(w'=w\,e^{-\lambda r}\)，此时 \(w=1\) 不再退化为恒等映射，`scale=1` 的断言会直接失败。
4. **零范数要靠「把 \(r\) 当 \(+\infty\)」处理**：\(\lambda r\to+\infty\) 时 \(e^{-\lambda r}\to 0\)，自然得到 \(w'=1\)。若直接算 \(0/0\) 得到 `NaN`，或算出 `inf` 再乘 `(w-1)` 得到 `inf`/`NaN`，都会污染整条采样轨迹。
5. **\(\lambda=0\) 必须精确退化为 CFG**：\(e^{0}=1\)，\(w'=1+(w-1)=w\)。这是最容易做对也最容易写错（比如把系数写成 \(w\,e^{-\lambda r}\)）的边界。
6. **方向完全不变**：MAMBO-G 只改尺度，输出的方向恒为 \(\pm\delta\) 之外仍以 \(\epsilon_u\) 为基准的直线。因此 \((out-u)/(c-u)\) 在每个样本内是常数，这条性质判题专门检查。

## 对拍口径

参考解在 float64 下用 `torch.linalg.vector_norm` 求两个逐样本范数，用布尔掩码把零范数样本的衰减因子置 0，最后转 float32。判题 `_oracle` 逐样本展平成 python 列表，用标量循环重算两个范数与 `math.exp`（`norm_uncond == 0` 时直接取 \(w'=1\)），不复用任何张量归约。结构断言五组：\(\lambda=0\) 等于裸 CFG、\(w=1\) 的逐位恒等、\((out-u)/(c-u)\) 的逐样本常数性并等于解析式的 \(w'\)、\(\lambda\) 增大时 \(w'\) 单调减小并收敛到 1、零范数样本无 `NaN`。阈值 `allclose(atol=1e-5)`。

## 复杂度

两遍逐元素读取（求两个范数）加一遍写出，\(O(N)\)，无额外缓冲；范数在除 batch 外的维度上归约。
