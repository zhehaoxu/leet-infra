# 详解：FDG 的逐频层解耦引导

## 出处

*Frequency-Decoupled Guidance*（arXiv:2506.19713）。实现口径对齐 HuggingFace diffusers 的 `src/diffusers/guiders/frequency_decoupled_guidance.py`：

- `forward`（L241–288）：逐层（高频 → 低频）做频带空间里的 CFG，最后用 `build_image_from_pyramid` 反变换；
- `project`（L42–60）：把 `shift` 相对 `v1 = pred_cond_freq` 做平行/正交分解，`v1` 先按除第 0 维外的全部维度归一化；
- `_is_fdg_enabled_for_level`（L319–335）：每层自己的窗口与尺度退化判定；
- `build_image_from_pyramid`（L63–72）：`img = pyrup(img) + pyramid[i]`，逐层上采样模糊后相加。

该 guider 依赖 kornia 的 `pyrup` 与 `build_laplacian_pyramid`。本题把金字塔拆解/重建当作外部前提，输入直接给同形状的频带，于是反变换就是「逐层相加」——既保留了 FDG 的全部数值结构（逐层尺度、投影、rescale），又不引入对第三方几何库的依赖。

## 核心与易错点

1. **平行方向的条件是「本层的条件预测」**：`project(shift, pred_cond_freq)`，即 \(\hat c_l=c_l/\lVert c_l\rVert\)。用 \(u_l\) 或直接对 `shift` 自身归一化都变成别的算子。
2. **投影只在 \(w_l\ne1\) 时做**：库源码用 `if not math.isclose(parallel_weight, 1.0)` 跳过。这一判断在数值上等价于「\(w=1\) 时 shift 不变」，但它保证了默认路径与裸 CFG 逐位一致。
3. **rescale 用的是无偏标准差**：`rescale_noise_cfg`（`guider_utils.py` L390–396）里是 `std(dim=list(range(1, ndim)), keepdim=True)`，`torch.std` 的默认 `correction=1`。写成总体标准差会让 \(\varphi=1\) 的「拉回条件尺度」断言差一个 \(\sqrt{n/(n-1)}\) 因子。
4. **逐层 rescale 在频带空间做，不是最后统一做一次**：`guidance_rescale_space == "freq"` 时每层各自 `rescale_noise_cfg(pred, pred_cond_freq, φ_l)`。若改成最后对求和结果 rescale 一次，各层的相对幅度关系就被抹平了。
5. **关闭的层透传它自己的条件层**：上游源码里 `else` 分支写的是 `pred_guided_pyramid.append(pred_cond_freq)`，而 `pred_cond_freq` 是循环内上一层留下的变量——这是一个真实的陈旧变量缺陷（关闭第 \(l\) 层时取到的是第 \(l-1\) 层的条件预测）。本题按语义修正为「透传本层的 `cond_levels[l]`」，判题也用修正后的口径。
6. **同尺度会塌缩成一次 CFG**：所有 \(s_l\) 相同、\(w_l\equiv1\)、\(\varphi_l\equiv0\) 时，\(\sum_l[u_l+s(c_l-u_l)]=\sum_l u_l+s(\sum_l c_l-\sum_l u_l)\)。这条恒等式是判题里最强的结构断言，任何「层间错位」都会立刻暴露。
7. **零条件层的 eps 是 1e-12**：`F.normalize` 的默认值。某层条件预测为零向量时 \(\hat c_l\) 取零，整个 `shift` 都算正交分量（被 \(w_l\) 完全保留）。

## 对拍口径

参考解在 float64 下按层循环：`torch.linalg.vector_norm` + 广播内积做投影，`torch.std(correction=1)` 做逐层 rescale，最后逐层累加转 float32。判题 `_oracle` 不复用任何张量归约——逐层 reshape 成 `(batch, width)`，用 python 标量重算 L2 范数、内积、平行/正交分解、无偏标准差与线性混合。结构断言六组：同尺度塌缩为整体 CFG、\(w=0\) 的正交性与 \(w=0.5\) 的线性关系、\(\varphi=1\) 时标准差等于条件预测且 \(\varphi\) 是线性混合系数、关闭层透传自身条件层（含三层中间层关闭）、四种参数组合与 oracle 逐值对拍、非法输入。阈值 `allclose(atol=1e-5)`。

## 复杂度

设总元素量为 \(N\)、层数为 \(L\)：每层一次投影（两次归约）加一次 rescale（两次标准差），总 \(O(N)\)，额外占用一个输出缓冲。
