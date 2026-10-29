# 详解：CFG-Zero* 的零初始化与最优尺度

## 出处

*CFG-Zero\*: Improved Classifier-Free Guidance for Flow Matching Models*（arXiv:2503.18886）。实现口径对齐 HuggingFace diffusers 的 `src/diffusers/guiders/classifier_free_zero_star_guidance.py`：`forward`（L100–124）给出「零初始化窗口 → 尺度校正 → 外推」的分支，`cfg_zero_star_scale`（L156–164）给出 \(\alpha=\langle c,u\rangle/(\lVert u\rVert^2+\varepsilon)\) 与 `eps=1e-8`。论文建议零初始化覆盖前 4% 的推理步，diffusers 默认 `zero_init_steps=1`。

## 核心与易错点

1. **\(\alpha\) 是最小二乘尺度，不是余弦相似度**：分母是 \(\lVert\epsilon_u\rVert^2\)，不是 \(\lVert\epsilon_c\rVert\lVert\epsilon_u\rVert\)。写成余弦会丢掉「两支幅度差异」这一关键信息，残差正交的判题断言直接失败。
2. **窗口判定是严格小于**：`step < k` 才置零，第 \(k\) 步就要走引导。用 `<=` 会少做一步引导。
3. **窗口内返回的是全零张量**，不是 \(\epsilon_c\)、也不是上一步的预测——它相当于让采样器在该步不做任何更新。
4. **\(\varepsilon\) 加在分母的平方和上**，不乘在 \(\alpha\) 上，也不加在根号里；它只让 \(\lvert\alpha\rvert\) 变小。
5. **\(\alpha\) 允许为负**：两支反向时 \(\langle c,u\rangle<0\)，此时加 clamp 会得到完全不同的结果。
6. **外推基准是 \(\alpha\epsilon_u\)**：`out = ᾱε_u + s(ε_c − ᾱε_u)`，因此 \(s=1\) 时恰好回到 \(\epsilon_c\)，这是一个很好的自检点。
7. **逐样本一个标量**：归约必须覆盖除 batch 外的全部维度；对整个 batch 求一次内积会让不同样本互相污染。

## 对拍口径

参考解在 float64 下用 `torch.sum` 沿除第 0 维外的维度归约，最后转 float32。判题 `_oracle` 逐样本把张量展平成 python 列表，自己算内积与平方和（不复用张量归约）。结构断言五组：窗口边界（前 \(k\) 步全零、第 \(k\) 步恢复）、\(s=1\) 恒等、由输出反解 \(\alpha\epsilon_u\) 后验证残差正交、正交/平行两种退化、\(\varepsilon\) 增大时 \(\lvert\alpha\rvert\) 单调变小。阈值 `allclose(atol=1e-5)`。

## 复杂度

逐元素 \(O(N)\)，外加一次逐样本内积与平方和归约（也是 \(O(N)\)）。
