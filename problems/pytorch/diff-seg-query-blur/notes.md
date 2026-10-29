# 详解：SEG 的查询平滑

## 出处

*Smoothed Energy Guidance*（arXiv:2408.00760）。实现口径对齐 HuggingFace diffusers 的 `src/diffusers/hooks/smoothed_energy_guidance_utils.py`：`SmoothedEnergyGuidanceHook.post_forward` 先按 `kernel_size = ceil(6*sigma) + 1 - ceil(6*sigma) % 2` 定核长，再调用 `_gaussian_blur_2d`；后者用 `is_inf = sigma > sigma_threshold_inf` 分成「卷积」与「均值」两支，取前 `n*n` 个 token、`permute` 成 `(B, D, n, n)`，卷积支用 `F.pad(..., mode="reflect")` 加 `F.conv2d(groups=D)`，最后把结果写回切片。guider 侧的四分支合成（`forward` 与 `pred_cond_seg`）与 `diff-pag-dual-guidance`（order 504）的骨架逐字相同，属于重复机制，本题不重出，只出 SEG 真正独有的这一段。

## 核心与易错点

1. **「无限模糊」并不等于纯均匀平均**：`sigma > threshold_inf` 时走的是**卷积**，只是核长被夹到网格尺度、\(\sigma\) 极大使核近似常数，于是结果「接近」全局均值但**不等于**它——反射填充让边缘邻域被重复计入，示例里 3×3 网格上的结果是 2.666667 而不是 3.0。把这一支写成 `patch.mean()` 会在测试里被逐位抓住。
2. **核长先是奇数、再被夹取**：`ceil(6σ)+1−ceil(6σ)%2` 把 6σ 变成不小于它的最小奇数；随后 `min(k0, n − (n%2 − 1))`，注意上限是 \(n+1\)（\(n\) 偶）或 \(n\)（\(n\) 奇），不是简单的 \(n\)。写 `min(k0, n)` 在 \(n\) 为偶数时会漏掉一个可用的核长。
3. **填充方式是 reflect**：只有 `reflect` 不重复边缘；`replicate` 与零填充都会让边界行不同。判断标准是 `F.pad` 的 `mode`，而不是「看起来差不多」。
4. **填充宽度是 `k // 2`**：\(k\) 为奇数时对称；这一段不需要处理偶数 \(k\)（上游已凑奇）。
5. **特征图重排次序**：先 `permute(0, 2, 1)` 再 `reshape(B, D, n, n)`，写回时 `reshape(B, D, n*n).permute(0, 2, 1)`。少一次 permute 会让通道与 token 混在一起，结果仍然「形状正确」，但数值全错。
6. **尾巴 token 不动**：`S` 不是完全平方数时，最后 \(S-n^2\) 个 token 不在任何方格内，必须逐位保留原值。
7. **分组卷积的组数**：`groups=D`，卷积核形状 `(D, 1, k, k)`，每个通道独立卷积；若用 `groups=1` 且只给单通道核，会变成跨通道混合。

## 对拍口径

参考解 float64 计算、float32 返回。判题 `_oracle` 不用 `F.pad` / `F.conv2d`：反射索引用 python 手写的 `while` 折返实现，卷积用四重标量循环逐点累加，高斯核用 `math.exp` 从头构造，核长单独由 `_kernel_size` 重算。八组断言：示例两支、均值分支（逐 (b, d) 均值、通道间不同）、阈值严格性（\(\sigma=\text{thr}\) 走均值、\(\text{thr}<\sigma\) 走卷积）、核长夹取（\(n=6,\sigma=1.5\) 时 \(k=7\)，并与未夹取的 \(k=9\) 对比确认不同；\(n=3\) 时上限为 3）、尾巴 token 保留（含 \(S=1\)）、六组 (n, D, σ, thr) 的对拍与非方形 token 数、输入不被就地修改、非法输入。阈值 `atol=1e-5`。

## 复杂度

重排与写回是 \(O(BDn^2)\) 次搬运；卷积是 \(O(BDn^2k^2)\) 次乘加，\(k\le n+1\)，最坏 \(O(BDn^4)\)。核构造 \(O(k)\)。
